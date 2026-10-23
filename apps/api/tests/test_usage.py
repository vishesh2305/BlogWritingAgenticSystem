import asyncio

from cwas.models import Usage
from cwas.usage import UsageMeter, agent_scope, current_meter, metering, record_usage


def test_recording_without_an_active_meter_is_a_no_op() -> None:
    record_usage(10, 5)

    assert current_meter() is None


def test_records_totals_and_per_agent_usage() -> None:
    meter = UsageMeter()

    with metering(meter):
        with agent_scope("writer"):
            record_usage(100, 50)
            record_usage(10, 5)
        record_usage(1, 1)

    assert meter.total == Usage(calls=3, input_tokens=111, output_tokens=56)
    assert meter.by_agent == {"writer": Usage(calls=2, input_tokens=110, output_tokens=55)}
    assert current_meter() is None


async def test_meter_follows_calls_into_child_tasks() -> None:
    meter = UsageMeter()

    async def call() -> None:
        record_usage(3, 4)

    with metering(meter), agent_scope("fact_checker"):
        await asyncio.gather(call(), call())

    assert meter.total.calls == 2
    assert meter.by_agent["fact_checker"].output_tokens == 8
