#!/usr/bin/env python3
"""
Check the checkpoint for manual test session
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isa_agent_sdk.services.durable_service import durable_service


async def main():
    session_id = "manual-test-1761235093"  # From the most recent test

    print(f"Checking checkpoint for: {session_id}\n")

    checkpointer = durable_service.get_checkpointer()
    config = {"configurable": {"thread_id": session_id}}

    try:
        state_tuple = await checkpointer.aget_tuple(config)

        if state_tuple:
            print("✅ Checkpoint found!")
            checkpoint_data = state_tuple.checkpoint

            if 'channel_values' in checkpoint_data:
                channel_values = checkpoint_data['channel_values']
                messages = channel_values.get('messages', [])
                summary = channel_values.get('summary')

                print(f"\nMessages in checkpoint: {len(messages)}\n")

                for idx, msg in enumerate(messages):
                    msg_type = type(msg).__name__
                    content = getattr(msg, 'content', str(msg))
                    print(f"[{idx}] {msg_type}:")
                    print(f"    {content[:200]}")
                    print()

                if summary:
                    print(f"Summary: {summary[:200]}")
            else:
                print("❌ No channel_values in checkpoint")
        else:
            print("❌ No checkpoint found!")
            print("\nChecking if session exists in DB...")

            conn_pool = checkpointer.conn
            async with conn_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT COUNT(*) FROM checkpoints WHERE thread_id = %s",
                        (session_id,)
                    )
                    count = (await cur.fetchone())[0]
                    print(f"Rows in database for this session: {count}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
