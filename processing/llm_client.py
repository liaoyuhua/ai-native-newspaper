"""
OpenAI 兼容接口封装。

默认供应商：LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
便宜供应商（可选）：LLM_CHEAP_API_KEY / LLM_CHEAP_BASE_URL + LLM_CHEAP_MODEL / LLM_MID_MODEL

提供三类调用：
- chat_json:      普通对话，要求模型返回 JSON
- chat_text:      普通对话，返回纯文本
- run_tool_loop:  带工具调用的自主循环（深度研究 agent）
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from openai import OpenAI, RateLimitError, APIStatusError, APIConnectionError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

import config

logger = logging.getLogger(__name__)

_client_default: OpenAI | None = None
_client_cheap: OpenAI | None = None


def get_client(model: str | None = None) -> OpenAI:
    """按模型名选择默认档或便宜档 client。"""
    global _client_default, _client_cheap
    use_cheap = bool(model) and config.llm_uses_cheap_provider(model)

    if use_cheap:
        if _client_cheap is None:
            _client_cheap = OpenAI(
                api_key=config.LLM_CHEAP_API_KEY,
                base_url=config.LLM_CHEAP_BASE_URL,
                max_retries=0,
            )
        return _client_cheap

    if _client_default is None:
        if not config.LLM_API_KEY:
            raise RuntimeError(
                "缺少 LLM_API_KEY，请在 .env 中配置（同时设置 LLM_BASE_URL / LLM_MODEL）"
            )
        _client_default = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL, max_retries=0)
    return _client_default


def _request_kwargs(model: str, **kwargs: Any) -> dict[str, Any]:
    """合并调用方参数与对应供应商的 EXTRA_KWARGS / TEMPERATURE。调用方参数优先，再被强制 temperature 覆盖。"""
    if config.llm_uses_cheap_provider(model):
        extra = config.LLM_CHEAP_EXTRA_KWARGS
        forced_temp = config.LLM_CHEAP_TEMPERATURE
    else:
        extra = config.LLM_EXTRA_KWARGS
        forced_temp = config.LLM_TEMPERATURE

    merged = {**extra, **kwargs, "model": model}
    if forced_temp is not None:
        merged["temperature"] = forced_temp
    return merged


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        # 余额/配额耗尽不会随退避恢复，应立即交给显式 fallback，而不是重复付出延迟。
        text = str(exc).lower()
        return "exceeded_current_quota" not in text and "insufficient balance" not in text
    if isinstance(exc, APIConnectionError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code in {408, 409, 429, 500, 502, 503, 504}:
        return True
    return False


def _parse_json_content(content: str | None) -> dict[str, Any]:
    """容忍模型包一层 markdown 代码块，或前后夹杂杂质。"""
    text = (content or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].lstrip()
        if text.endswith("```"):
            text = text[:-3].rstrip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=3, min=5, max=30),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _create_completion(**kwargs: Any):
    model = str(kwargs.get("model") or "")
    return get_client(model).chat.completions.create(**kwargs)


def chat_json(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> dict[str, Any]:
    """调用模型并要求返回严格 JSON。"""
    last_error: json.JSONDecodeError | None = None
    for attempt in range(2):
        resp = _create_completion(
            **_request_kwargs(
                model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
        )
        content = resp.choices[0].message.content
        try:
            return _parse_json_content(content)
        except json.JSONDecodeError as exc:
            last_error = exc
            logger.warning("模型未返回合法 JSON（第 %d/2 次）：%s", attempt + 1, (content or "")[:500])
    assert last_error is not None
    raise last_error


def chat_json_with_fallback(
    *,
    model: str,
    fallback_model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> tuple[dict[str, Any], str]:
    """仅在主模型不可用时显式降级，并返回实际使用的模型名。"""
    try:
        return chat_json(model, system_prompt, user_prompt, temperature), model
    except (RateLimitError, APIStatusError, APIConnectionError) as exc:
        if not fallback_model or fallback_model == model:
            raise
        logger.warning("模型 %s 不可用，写作阶段降级到 %s: %s", model, fallback_model, exc)
        return chat_json(fallback_model, system_prompt, user_prompt, temperature), fallback_model


def chat_text(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.4,
) -> str:
    """调用模型，返回纯文本。"""
    resp = _create_completion(
        **_request_kwargs(
            model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
    )
    return resp.choices[0].message.content or ""


def run_tool_loop(
    model: str,
    system_prompt: str,
    user_prompt: str,
    tools: list[dict[str, Any]],
    tool_impls: dict[str, Callable[..., Any]],
    max_tool_calls: int,
    on_step: Callable[[str, dict, Any], None] | None = None,
) -> str:
    """
    ReAct 风格的自主工具调用循环。

    模型可以连续多轮调用 tools 里声明的工具；只要它还在发起 tool_calls 就继续循环，
    直到它给出一段不带 tool_calls 的最终文本回复，或达到 max_tool_calls 上限。
    """
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    calls_made = 0
    finish_prompted = False
    while True:
        force_finish = calls_made >= max_tool_calls
        call_kwargs: dict[str, Any] = {
            "messages": messages,
            "temperature": 0.3,
        }
        if not force_finish:
            call_kwargs["tools"] = tools
            call_kwargs["tool_choice"] = "auto"
        elif not finish_prompted:
            # 触达预算后强制收尾，避免模型继续空转
            messages.append(
                {
                    "role": "user",
                    "content": "工具调用次数已达上限。请立刻停止调用工具，基于已有证据输出最终研究档案 JSON。",
                }
            )
            finish_prompted = True

        resp = _create_completion(**_request_kwargs(model, **call_kwargs))
        msg = resp.choices[0].message

        if force_finish or not msg.tool_calls:
            return msg.content or ""

        messages.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        # 同一轮 assistant.tool_calls 必须全部回完，不能中途 break，
        # 否则下一轮请求会因缺少 tool 回复而 400。
        for tc in msg.tool_calls:
            calls_made += 1
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}

            impl = tool_impls.get(name)
            if impl is None:
                result: Any = {"error": f"未知工具: {name}"}
            else:
                try:
                    result = impl(**args)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("工具 %s 执行失败", name)
                    result = {"error": str(exc)}

            if on_step:
                on_step(name, args, result)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False)[:8000],
                }
            )
