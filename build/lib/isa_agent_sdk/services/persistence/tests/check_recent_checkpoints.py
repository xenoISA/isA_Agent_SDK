#!/usr/bin/env python3
"""
Check most recent checkpoints
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from isa_agent_sdk.services.durable_service import durable_service


async def main():
    checkpointer = durable_service.get_checkpointer()

    print("Checking recent checkpoints with 'test' pattern...")

    try:
        conn_pool = checkpointer.conn
        async with conn_pool.connection() as conn:
            async with conn.cursor() as cur:
                # Get thread_ids containing 'test'
                await cur.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE '%test%' LIMIT 20"
                )
                rows = await cur.fetchall()

                if rows:
                    print(f"\nFound {len(rows)} test-related thread_ids:")
                    for row in rows:
                        thread_id = row[0]
                        print(f"  - {thread_id}")

                        # Get checkpoint for this thread
                        config = {"configurable": {"thread_id": thread_id}}
                        state_tuple = await checkpointer.aget_tuple(config)

                        if state_tuple and hasattr(state_tuple, 'checkpoint'):
                            checkpoint_data = state_tuple.checkpoint
                            if 'channel_values' in checkpoint_data:
                                channel_values = checkpoint_data['channel_values']
                                messages = channel_values.get('messages', [])
                                print(f"    Messages: {len(messages)}")
                else:
                    print("\n❌ No test-related checkpoints")

                # Also check for session patterns
                print("\n\nChecking for 'session' pattern...")
                await cur.execute(
                    "SELECT DISTINCT thread_id FROM checkpoints WHERE thread_id LIKE '%session%' LIMIT 20"
                )
                rows = await cur.fetchall()

                if rows:
                    print(f"\nFound {len(rows)} session-related thread_ids:")
                    for row in rows:
                        print(f"  - {row[0]}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
