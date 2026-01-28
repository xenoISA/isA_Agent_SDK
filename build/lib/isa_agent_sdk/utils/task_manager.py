"""
Async Task Manager
Manages background tasks with proper lifecycle and cleanup
"""
import asyncio
import weakref
from typing import Set, Dict, Optional, Callable, Any
from datetime import datetime
from isa_agent_sdk.utils.logger import api_logger


class TaskManager:
    """Manages async tasks with proper cleanup"""
    
    def __init__(self):
        self.tasks: Set[asyncio.Task] = set()
        self.task_metadata: Dict[asyncio.Task, Dict[str, Any]] = {}
        self._shutdown_event = asyncio.Event()
        self._cleanup_interval = 30  # seconds
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_monitor()
    
    def create_task(self, coro, name: str = None, timeout: float = None) -> asyncio.Task:
        """Create and track a task"""
        task = asyncio.create_task(coro, name=name)
        
        # Track the task
        self.tasks.add(task)
        self.task_metadata[task] = {
            "name": name or f"task_{id(task)}",
            "created_at": datetime.now(),
            "timeout": timeout
        }
        
        # Add cleanup callback
        task.add_done_callback(self._task_done_callback)
        
        # Add timeout if specified
        if timeout:
            def timeout_callback():
                if not task.done():
                    task.cancel()
                    api_logger.warning(f"Task '{name}' timed out after {timeout}s")
            
            asyncio.get_event_loop().call_later(timeout, timeout_callback)
        
        api_logger.debug(f"Created task '{name or task.get_name()}' ({len(self.tasks)} active)")
        return task
    
    def _task_done_callback(self, task: asyncio.Task):
        """Callback when task completes"""
        if task in self.tasks:
            self.tasks.remove(task)
        
        metadata = self.task_metadata.pop(task, {})
        task_name = metadata.get("name", "unknown")
        
        if task.cancelled():
            api_logger.debug(f"Task '{task_name}' was cancelled")
        elif task.exception():
            api_logger.error(f"Task '{task_name}' failed: {task.exception()}")
        else:
            api_logger.debug(f"Task '{task_name}' completed successfully")
    
    def _start_cleanup_monitor(self):
        """Start background cleanup monitor"""
        async def cleanup_monitor():
            while not self._shutdown_event.is_set():
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), 
                        timeout=self._cleanup_interval
                    )
                except asyncio.TimeoutError:
                    # Periodic cleanup
                    await self._cleanup_completed_tasks()
        
        self._cleanup_task = asyncio.create_task(cleanup_monitor())
    
    async def _cleanup_completed_tasks(self):
        """Clean up completed tasks"""
        completed_tasks = [task for task in self.tasks if task.done()]
        
        for task in completed_tasks:
            self._task_done_callback(task)
        
        if completed_tasks:
            api_logger.debug(f"Cleaned up {len(completed_tasks)} completed tasks")
    
    async def cancel_all_tasks(self, timeout: float = 5.0):
        """Cancel all running tasks"""
        if not self.tasks:
            return
        
        api_logger.info(f"Cancelling {len(self.tasks)} running tasks...")
        
        # Cancel all tasks
        for task in self.tasks.copy():
            if not task.done():
                task.cancel()
        
        # Wait for cancellation with timeout
        if self.tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*self.tasks, return_exceptions=True),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                api_logger.warning(f"Some tasks did not cancel within {timeout}s timeout")
        
        # Force cleanup
        self.tasks.clear()
        self.task_metadata.clear()
        
        api_logger.info("All tasks cancelled and cleaned up")
    
    async def shutdown(self):
        """Shutdown task manager"""
        self._shutdown_event.set()
        
        # Cancel cleanup monitor
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Cancel all managed tasks
        await self.cancel_all_tasks()
        
        api_logger.info("Task manager shutdown complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get task statistics"""
        running_tasks = [task for task in self.tasks if not task.done()]
        completed_tasks = [task for task in self.tasks if task.done()]
        
        task_details = []
        for task in running_tasks:
            metadata = self.task_metadata.get(task, {})
            task_details.append({
                "name": metadata.get("name", "unknown"),
                "created_at": metadata.get("created_at"),
                "state": "running" if not task.done() else "done",
                "cancelled": task.cancelled(),
                "has_exception": task.done() and task.exception() is not None
            })
        
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": len(running_tasks),
            "completed_tasks": len(completed_tasks),
            "task_details": task_details
        }


# Global task manager
task_manager = TaskManager()


# Context manager for automatic task cleanup
class ManagedTaskContext:
    """Context manager for automatic task cleanup"""
    
    def __init__(self, name: str = None):
        self.name = name
        self.tasks: Set[asyncio.Task] = set()
    
    def create_task(self, coro, name: str = None, timeout: float = None) -> asyncio.Task:
        """Create task in this context"""
        task_name = f"{self.name}.{name}" if self.name and name else (name or self.name)
        task = task_manager.create_task(coro, task_name, timeout)
        self.tasks.add(task)
        return task
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cancel all tasks in this context
        for task in self.tasks.copy():
            if not task.done():
                task.cancel()
        
        # Wait for cancellation
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        self.tasks.clear()


# Convenience functions
def create_managed_task(coro, name: str = None, timeout: float = None) -> asyncio.Task:
    """Create a managed task"""
    return task_manager.create_task(coro, name, timeout)


async def run_with_timeout(coro, timeout: float, name: str = None):
    """Run coroutine with timeout using managed task"""
    task = create_managed_task(coro, name, timeout)
    try:
        return await task
    except asyncio.CancelledError:
        raise asyncio.TimeoutError(f"Task '{name}' timed out after {timeout}s")