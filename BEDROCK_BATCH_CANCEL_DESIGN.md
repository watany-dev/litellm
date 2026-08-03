# Bedrock batch cancel (StopModelInvocationJob) 対応

## Context

Bedrockバッチジョブに対する `POST /v1/batches/{batch_id}/cancel` がHTTP 400を返す。原因は `litellm/batches/main.py` の `cancel_batch()` (L869) がopenai/azure/vertex_aiのハードコードif/elifのみで、bedrockは `else` (L993-1005) の `BadRequestError` に落ちるため。create/retrieveに存在する `ProviderConfigManager.get_provider_batches_config` -> `base_llm_http_handler` の汎用分岐がcancelには存在せず、`BaseBatchesConfig` にもcancel用メソッドがない

### AWS仕様 (確認済み)

- `StopModelInvocationJob`: `POST /model-invocation-job/{jobIdentifier}/stop`、リクエストボディなし、成功時HTTP 200・空ボディ
- jobIdentifierはジョブARN可(URLではパーセントエンコード)。IAM: `bedrock:StopModelInvocationJob`
- エラー: AccessDenied / Conflict(停止不能状態) / ResourceNotFound / Throttling / ValidationException
- stop後のstatusは `Stopping` -> `Stopped`。既存マップ `_BEDROCK_MIJ_STATUS_TO_OPENAI` (litellm/llms/bedrock/batches/handler.py:12-23) が `Stopping->cancelling`, `Stopped->cancelled` を既にカバー

### ユーザー決定事項

- 実装経路: **BaseBatchesConfig抽象 + SigV4/httpx**(retrieveのprovider-config経路のミラー。boto3短絡ではない)
- レスポンス: **stop成功後にGETで再取得**し、既存のretrieve変換で正確なstatusを返す(Vertexのcancelと同パターン)

## 変更ファイル

1. `litellm/llms/base_llm/batches/transformation.py` — cancel抽象の追加
2. `litellm/llms/bedrock/batches/transformation.py` — bedrock実装
3. `litellm/llms/custom_httpx/llm_http_handler.py` — `cancel_batch` / `async_cancel_batch` 追加
4. `litellm/batches/main.py` — provider-config分岐の追加、型ヒント
5. `litellm/proxy/batches_endpoints/endpoints.py` — cancelシナリオ1に `data["model"]` 注入

## 実装ステップ

### Step 1: BaseBatchesConfig にcancelサーフェスを追加

`litellm/llms/base_llm/batches/transformation.py` に **非abstract** メソッドを追加(既存プロバイダのconfigに実装を強制しないため。デフォルトは `NotImplementedError` raise):

```python
@property
def should_retrieve_batch_after_cancel(self) -> bool:
    return False

def transform_cancel_batch_request(
    self, batch_id: str, optional_params: dict, litellm_params: dict,
) -> Union[bytes, str, Dict[str, Any]]:
    raise NotImplementedError(...)

def transform_cancel_batch_response(
    self, model: Optional[str], raw_response: httpx.Response,
    logging_obj: LiteLLMLoggingObj, litellm_params: dict,
) -> LiteLLMBatch:
    raise NotImplementedError(...)
```

シグネチャは既存の `transform_retrieve_batch_request` (L162) / `transform_retrieve_batch_response` (L182) を正確にミラーする

### Step 2: BedrockBatchesConfig にcancel変換を実装

`litellm/llms/bedrock/batches/transformation.py`:

- `transform_cancel_batch_request`: `transform_retrieve_batch_request` (L347-402) をテンプレートに:
  - ARN検証(`arn:aws:bedrock:` prefix、パーツ数、region正規表現)は retrieve と同一ロジック。共通部分はプライベートヘルパー(例: `_validate_and_encode_job_arn(batch_id) -> tuple[str, str]` がregionとencoded_arnを返す)に抽出してretrieve側もそれを使うようリファクタ
  - `":async-invoke/" in batch_id` の場合は明示的に `ValueError("Cancellation is not supported for Bedrock async-invoke jobs")`(bedrock-runtimeデータプレーンにstop APIが存在しないため)
  - URL: `https://bedrock.{region}.amazonaws.com/model-invocation-job/{encoded_arn}/stop`
  - `self.common_utils.sign_aws_request(service_name="bedrock", data="", endpoint_url=..., optional_params=optional_params, method="POST")` で署名。**注意**: 署名ペイロードと実送信ボディの一致が必須。`sign_aws_request` (common_utils.py:1441-1455) はPOSTで `data=""` を渡すと `request_data=""` で署名し `b""` を返すので、pre-signed dictは `{"method": "POST", "url": ..., "headers": signed_headers, "data": signed_data}`(空bytes)とする。StopModelInvocationJobはボディなしなので空ボディで送る
- `should_retrieve_batch_after_cancel` プロパティを `True` でオーバーライド(stopの200は空ボディなので、レスポンスはretrieveフローの再利用で構築する。`transform_cancel_batch_response` はbedrockでは呼ばれないため実装しない)

### Step 3: BaseLLMHTTPHandler に cancel_batch / async_cancel_batch

`litellm/llms/custom_httpx/llm_http_handler.py`、`retrieve_batch` (L3891-3975) / `async_retrieve_batch` (L4056-) を正確にミラー:

```python
def cancel_batch(self, batch_id, litellm_params, provider_config, headers,
                 api_base, api_key, logging_obj, _is_async=False, client=None,
                 timeout=None, model=None) -> Union[LiteLLMBatch, Coroutine[...]]:
```

フロー:
1. `transformed_request = provider_config.transform_cancel_batch_request(batch_id=..., optional_params=litellm_params, litellm_params=litellm_params)`
2. `_is_async` なら `async_cancel_batch` に委譲(coroutine返却)
3. pre-signed dict分岐(`"method" in transformed_request`)は retrieve L3935-3947 と同じディスパッチ。POSTなので `data` を付与
4. 例外は `self._handle_error(e, provider_config)`(retrieve同様)。ConflictException等のAWSエラーは非2xx -> `HTTPStatusError` -> `get_error_class` -> `BedrockError` 経由で適切な4xxにマップされる(既存機構で追加実装不要)
5. 成功後: `provider_config.should_retrieve_batch_after_cancel` なら `return self.retrieve_batch(batch_id=..., ..., _is_async=False)` で再取得(async版では `_is_async=True` で返るcoroutineをawait)。そうでなければ `provider_config.transform_cancel_batch_response(...)`

### Step 4: main.py cancel_batch に provider-config 分岐

`litellm/batches/main.py`:

- `cancel_batch()` 内、`_is_async` 取得(L921)の後・openai分岐(L923)の前に、retrieveのL590-623をミラーした分岐を挿入:

```python
if model is not None:
    provider_config = ProviderConfigManager.get_provider_batches_config(
        model=model, provider=LlmProviders(custom_llm_provider))
else:
    provider_config = None
if provider_config is not None:
    return base_llm_http_handler.cancel_batch(
        batch_id=batch_id, provider_config=provider_config,
        litellm_params=litellm_params, headers=extra_headers or {},
        api_base=optional_params.api_base, api_key=optional_params.api_key,
        logging_obj=kwargs.get("litellm_logging_obj") or LiteLLMLoggingObj(
            model=model or f"{custom_llm_provider}/unknown", messages=[], stream=False,
            call_type="batch_cancel", start_time=None,
            litellm_call_id="batch_cancel_" + batch_id, function_id="batch_cancel"),
        _is_async=_is_async,
        client=(client if isinstance(client, (HTTPHandler, AsyncHTTPHandler)) else None),
        timeout=timeout, model=model)
```

  注: `cancel_batch` は既に `model` を引数に持つ(L871)のでkwargsからではなく引数を使う。`client = kwargs.get("client", None)` の取得も必要(retrieve L558と同様)
- 型ヒント: `acancel_batch` (L826) と `cancel_batch` (L872) の `Literal` に `"bedrock"` を追加
- `else` のエラーメッセージ (L995) を retrieve の L489-494 と同様に更新: bedrockは `model` kwargs必須である旨を明記("'bedrock' is supported but requires `model` to be passed so the provider config can be loaded")

### Step 5: proxy cancelエンドポイントで model を渡す

`litellm/proxy/batches_endpoints/endpoints.py` のcancelシナリオ1 (L868-894): retrieveのL496-499と同じ修正を適用 — `data["batch_id"] = ...` (L882) の直後に `data["model"] = model_from_id` を追加。これがないとSDKのprovider-config分岐に到達せずレガシー分岐で400になる

シナリオ2(router経由)は `router._acancel_batch` (router.py:5382-5414) がdeployment litellm_paramsの `model` をspreadで渡すため修正不要。シナリオ3(生ARN + `?provider=bedrock`、modelなし)はretrieveと同様に「modelが必要」という明確な400メッセージになる(Step 4のメッセージ改善でカバー)

## リクエストトレース(変更後)

`POST /v1/batches/{encoded_id}/cancel` -> proxy `cancel_batch` (endpoints.py:799) -> シナリオ1で `data["model"]` 注入 or シナリオ2でrouterがmodelを渡す -> `litellm.acancel_batch` -> executor -> `cancel_batch` -> **新provider-config分岐** -> `base_llm_http_handler.cancel_batch` -> `BedrockBatchesConfig.transform_cancel_batch_request`(SigV4署名済み `POST .../model-invocation-job/{encoded_arn}/stop`)-> 200空ボディ -> `should_retrieve_batch_after_cancel=True` なので `retrieve_batch` で再GET -> `transform_retrieve_batch_response` -> `LiteLLMBatch(status="cancelling"|"cancelled")` -> `update_batch_in_database(operation="cancel")` -> 200

## テスト計画

- `tests/test_litellm/llms/bedrock/batches/test_transformation.py`(既存ファイルに追加):
  - `transform_cancel_batch_request`: URLが `/stop` で終わりmethodがPOST、ARNが完全エンコードされている、署名ヘッダ(Authorization等)が存在、`data` が署名ペイロードと一致(空)
  - async-invoke ARN -> ValueError、不正ARN/不正region -> ValueError
  - `should_retrieve_batch_after_cancel` が True
- `tests/test_litellm/batches/test_main.py`(既存cancelセクションに追加):
  - **回帰テスト(修正前は失敗)**: `cancel_batch(batch_id="arn:aws:bedrock:us-west-2:...:model-invocation-job/abc", custom_llm_provider="bedrock", model="bedrock/us.anthropic.claude-...")` で `base_llm_http_handler.cancel_batch` がモック経由で呼ばれ、`BadRequestError` が出ないこと(現状はL993で400)
  - `acancel_batch` のbedrock委譲(既存の delegation テストのミラー)
  - bedrockで `model` なし -> BadRequestError(メッセージに model 必須の旨)
- `tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`:
  - `cancel_batch`: pre-signed dict のPOSTディスパッチ、200後に retrieve が呼ばれ LiteLLMBatch が返る(`should_retrieve_batch_after_cancel=True` のconfigで)、非2xx -> `_handle_error` 経由の例外
  - `should_retrieve_batch_after_cancel=False` のconfigでは `transform_cancel_batch_response` が呼ばれる
- `tests/test_litellm/proxy/batches_endpoints/test_endpoints.py`: cancelシナリオ1で `litellm.acancel_batch` に `model=model_from_id` が渡ること

既存のbedrock retrieveテスト(test_transformation.py / test_handler.py)はヘルパー抽出リファクタの回帰網として全パス維持

## エッジケース

- gov-cloud/China ARN (`arn:aws-us-gov:` 等): 既存retrieveの `arn:aws:bedrock:` literal prefixチェックが拒否する既知の制限。cancelもretrieveと同一ヘルパーを使い挙動を一致させる(この修正のスコープ外だが、ヘルパー化により将来1箇所で直せる)
- ジョブが終端状態: AWSが `ConflictException` -> 既存 `get_error_class` 経由で4xx(500にならない)
- 空ボディ署名: 署名時と送信時のペイロード一致をテストで担保(不一致は403 SignatureDoesNotMatchになる)

## 検証(QA runbook)

1. `python litellm/proxy/proxy_cli.py --config litellm/proxy/dev_config.yaml --detailed_debug --reload --use_v2_migration_resolver 2>&1 | tee litellm.log` でproxy起動(bedrockモデルとAWS認証はdev_config/.env)
2. ファイルアップロード + バッチ作成(実Bedrock API): `curl http://localhost:4000/v1/files -H "Authorization: Bearer sk-1234" -F purpose=batch -F file=@batch.jsonl` -> `curl http://localhost:4000/v1/batches -H "Authorization: Bearer sk-1234" -d '{"input_file_id": "...", "endpoint": "/v1/chat/completions", "completion_window": "24h", "model": "<bedrock model>"}'`
3. 返ったbatch idで即キャンセル: `curl -X POST http://localhost:4000/v1/batches/{batch_id}/cancel -H "Authorization: Bearer sk-1234"` -> 200で `"status": "cancelling"` を確認(修正前は400)
4. `curl http://localhost:4000/v1/batches/{batch_id}` で `cancelling` -> `cancelled` 遷移を確認
5. 終端後にもう一度cancel -> ConflictException由来の4xxが返ることを確認
6. ユニットテスト: `pytest tests/test_litellm/llms/bedrock/batches/ tests/test_litellm/batches/test_main.py tests/test_litellm/llms/custom_httpx/test_llm_http_handler.py`、コミット前に `make pre-commit`
