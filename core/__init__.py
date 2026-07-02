"""
InversionAI - Centralized Event Logging System

Captures all activity from agents, skills, tools, and workflows
and makes it available for the UI log panel in real-time.
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from collections import deque


class LogSource(Enum):
    """Source category for log events."""
    AGENT = "agent"
    SKILL = "skill"
    TOOL = "tool"
    WORKFLOW = "workflow"
    DATA = "data"
    SYSTEM = "system"


class LogLevel(Enum):
    """Severity level for log events."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"


@dataclass
class LogEvent:
    """A single log event with full context."""
    timestamp: datetime
    source: LogSource
    level: LogLevel
    component: str  # e.g., "TomofastSkill", "WorkflowPlanner", "validate_data_file"
    message: str
    details: Optional[dict] = None

    def format_short(self) -> str:
        """Format as a concise one-line log entry."""
        ts = self.timestamp.strftime("%H:%M:%S.%f")[:-3]
        icon = self._level_icon()
        return f"[{ts}] {icon} [{self.source.value.upper()}:{self.component}] {self.message}"

    def format_detailed(self) -> str:
        """Format with full details."""
        line = self.format_short()
        if self.details:
            for key, value in self.details.items():
                line += f"\n         ├─ {key}: {value}"
        return line

    def _level_icon(self) -> str:
        icons = {
            LogLevel.DEBUG: "🔍",
            LogLevel.INFO: "ℹ️",
            LogLevel.WARNING: "⚠️",
            LogLevel.ERROR: "❌",
            LogLevel.SUCCESS: "✅",
        }
        return icons.get(self.level, "•")


class EventLogger:
    """
    Centralized event logger that captures all InversionAI activity.

    Thread-safe singleton that stores events in memory for UI display
    and optionally writes to a file.

    Usage:
        from core.logger import event_logger

        event_logger.log_agent("Orchestrator", "Processing user request", details={"intent": "gravity_inversion"})
        event_logger.log_skill("TomofastSkill", "Validating input data")
        event_logger.log_tool("validate_data_file", "Checking gravity data format")
    """

    _instance: Optional["EventLogger"] = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, max_events: int = 5000):
        if self._initialized:
            return
        self._initialized = True
        self._events: deque[LogEvent] = deque(maxlen=max_events)
        self._event_lock = threading.Lock()
        self._python_logger = logging.getLogger("inversionai")
        self._callbacks: list = []

    def log(
        self,
        source: LogSource,
        component: str,
        message: str,
        level: LogLevel = LogLevel.INFO,
        details: Optional[dict] = None,
    ) -> LogEvent:
        """Log an event from any source."""
        event = LogEvent(
            timestamp=datetime.now(),
            source=source,
            level=level,
            component=component,
            message=message,
            details=details,
        )
        with self._event_lock:
            self._events.append(event)

        # Also log to Python's logging system
        py_level = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.SUCCESS: logging.INFO,
        }.get(level, logging.INFO)
        self._python_logger.log(py_level, f"[{source.value}:{component}] {message}")

        # Notify callbacks
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception:
                pass

        return event

    # --- Convenience methods per source ---

    def log_agent(self, component: str, message: str, level: LogLevel = LogLevel.INFO, details: Optional[dict] = None):
        """Log an agent event (orchestrator, planner, routing)."""
        return self.log(LogSource.AGENT, component, message, level, details)

    def log_skill(self, component: str, message: str, level: LogLevel = LogLevel.INFO, details: Optional[dict] = None):
        """Log a skill event (setup, validate, configure, run)."""
        return self.log(LogSource.SKILL, component, message, level, details)

    def log_tool(self, component: str, message: str, level: LogLevel = LogLevel.INFO, details: Optional[dict] = None):
        """Log a tool invocation event."""
        return self.log(LogSource.TOOL, component, message, level, details)

    def log_workflow(self, component: str, message: str, level: LogLevel = LogLevel.INFO, details: Optional[dict] = None):
        """Log a workflow event (step transitions, completion)."""
        return self.log(LogSource.WORKFLOW, component, message, level, details)

    def log_data(self, component: str, message: str, level: LogLevel = LogLevel.INFO, details: Optional[dict] = None):
        """Log a data operation event (load, validate, convert)."""
        return self.log(LogSource.DATA, component, message, level, details)

    def log_system(self, component: str, message: str, level: LogLevel = LogLevel.INFO, details: Optional[dict] = None):
        """Log a system event (startup, config, errors)."""
        return self.log(LogSource.SYSTEM, component, message, level, details)

    # --- Query methods ---

    def get_all(self) -> list[LogEvent]:
        """Get all logged events."""
        with self._event_lock:
            return list(self._events)

    def get_recent(self, n: int = 50) -> list[LogEvent]:
        """Get the N most recent events."""
        with self._event_lock:
            events = list(self._events)
            return events[-n:]

    def get_by_source(self, source: LogSource, n: int = 50) -> list[LogEvent]:
        """Get events filtered by source."""
        with self._event_lock:
            filtered = [e for e in self._events if e.source == source]
            return filtered[-n:]

    def get_by_level(self, level: LogLevel, n: int = 50) -> list[LogEvent]:
        """Get events filtered by level."""
        with self._event_lock:
            filtered = [e for e in self._events if e.level == level]
            return filtered[-n:]

    def get_errors(self) -> list[LogEvent]:
        """Get all error events."""
        return self.get_by_level(LogLevel.ERROR, n=500)

    def get_formatted(self, n: int = 50, detailed: bool = False) -> str:
        """Get formatted log output as a string."""
        events = self.get_recent(n)
        if detailed:
            return "\n".join(e.format_detailed() for e in events)
        return "\n".join(e.format_short() for e in events)

    def clear(self):
        """Clear all logged events."""
        with self._event_lock:
            self._events.clear()

    def count(self) -> int:
        """Get total number of logged events."""
        return len(self._events)

    def add_callback(self, callback):
        """Add a callback that fires on each new event."""
        self._callbacks.append(callback)

    def remove_callback(self, callback):
        """Remove a callback."""
        self._callbacks = [cb for cb in self._callbacks if cb != callback]


# Singleton instance
event_logger = EventLogger()
