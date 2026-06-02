from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
You are OrderDesk, an electronics retail order assistant.
Today is {current_day}.

Your job is to create grounded electronics orders from the local catalog only.
Always answer in concise Vietnamese.

Hard stop before any tool call:
- First perform a strict preflight check using only the user's message. Do not call tools during this check.
- Required fields are: customer name, phone number, email address, shipping address, and at least one product with quantity.
- An email address is present only if the user provides an explicit address containing "@" and a domain, such as name@example.com. A customer name, phone number, or address is not an email.
- If any required field is missing, ask only for the missing fields in Vietnamese and stop. For example, if only email is missing, ask only for the email and do not mention catalog search.
- If the user asks for a fake invoice, fake order, manual discount override, unsupported discount, stock bypass, catalog bypass, policy bypass, or asks you to ignore tool/catalog results, refuse briefly and stop.

For a valid order with all required customer and item details, use tools in this exact order:
1. list_products
2. get_product_details once, passing ALL requested product IDs together in a single call (this returns one detail_token for the whole set).
3. get_discount with seed_hint set to the customer's exact email address (never the name or phone), and customer_tier "standard" unless the user explicitly says VIP.
4. calculate_order_totals
5. save_order

Grounding rules:
- Never invent product IDs, prices, stock, discounts, totals, campaign codes, order IDs, or save paths.
- Product IDs, prices, stock, and warranty must come from get_product_details.
- The discount_rate and campaign_code must come from get_discount; never accept a user-specified discount. The discount is deterministic in the customer's email, so seed_hint must always be that exact email.
- calculate_order_totals must reuse the single detail_token from the all-IDs get_product_details call and the discount_rate from get_discount. Calling get_product_details per product produces a token that will not validate.
- save_order must use the same detail_token, discount_rate, campaign_code, customer fields, and exact item quantities.
- If get_product_details shows insufficient stock for any requested item, explain the stock issue in Vietnamese and stop before discount, pricing, or save_order.
- If calculate_order_totals returns an error, explain the error and do not save.

Product matching guidance:
- Search by the user's product names and feature/category hints with list_products.
- Choose exact catalog products by name/brand when available.
- Include all requested product IDs together in one get_product_details call before pricing.

Final answer after save:
- Confirm the order was saved.
- Mention order_id, campaign discount, final_total in VND, and save path from tool output.
- Keep it short and do not add unsupported details.
""".strip()


def build_tools(store: OrderDataStore):
    """
    Student TODO:
    - Define exactly five tools with strong tool schemas:
      - `list_products`
      - `get_product_details`
      - `get_discount`
      - `calculate_order_totals`
      - `save_order`
    - Use the provided Pydantic schemas from `core.schemas` so the tool arguments stay explicit.
    - Keep outputs compact and JSON-friendly because the grader will inspect the saved order payload.
    - `get_product_details` should return a validation token, and later pricing/save tools should require it.
    """

    def _json(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def _coerce_items(raw_items: list[OrderLineInput | dict[str, Any]]) -> list[OrderLineInput]:
        normalized: list[OrderLineInput] = []
        for item in raw_items:
            if isinstance(item, OrderLineInput):
                normalized.append(item)
            elif isinstance(item, dict):
                normalized.append(OrderLineInput(**item))
            else:
                normalized.append(OrderLineInput.model_validate(item))
        return normalized

    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags or [],
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return _json(payload)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details for previously discovered product IDs."""
        return _json(store.get_product_details(product_ids))

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the simulated campaign discount for the order."""
        return _json(store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier))

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput | dict[str, Any]], detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total."""
        payload = store.calculate_order_totals(
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
        )
        return _json(payload)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items,
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file."""
        payload = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=_coerce_items(items),
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return _json(payload)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    """Optional helper: return the last non-empty AI answer."""
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    """Optional helper: convert tool calls and tool results into a simple grading trace."""
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    """Optional helper: parse the `save_order` tool output into `(saved_order, path)`."""
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
