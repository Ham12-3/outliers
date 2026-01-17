"""
Job scheduler for automated detection runs.

Uses APScheduler for scheduling daily detection jobs.
"""
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from .config import get_settings
from .database import SessionLocal
from .services.detection_pipeline import run_detection_pipeline

logger = logging.getLogger(__name__)
settings = get_settings()

# Global scheduler instance
_scheduler: BackgroundScheduler = None


def init_scheduler():
    """Initialise and start the scheduler."""
    global _scheduler

    if _scheduler is not None:
        return

    _scheduler = BackgroundScheduler(
        timezone="UTC",
        job_defaults={
            "coalesce": True,  # Combine missed runs into one
            "max_instances": 1,  # Only one instance of each job
            "misfire_grace_time": 3600,  # Allow 1 hour grace for misfires
        },
    )

    # Schedule daily detection run
    _scheduler.add_job(
        _run_detection_job,
        trigger=CronTrigger(
            hour=settings.detection_schedule_hour,
            minute=settings.detection_schedule_minute,
        ),
        id="daily_detection",
        name="Daily Anomaly Detection",
        replace_existing=True,
    )

    _scheduler.start()
    logger.info(
        f"Scheduler started. Daily detection scheduled at "
        f"{settings.detection_schedule_hour:02d}:{settings.detection_schedule_minute:02d} UTC"
    )


def shutdown_scheduler():
    """Shutdown the scheduler."""
    global _scheduler

    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
        logger.info("Scheduler shutdown")


def _run_detection_job():
    """Job function to run detection pipeline."""
    logger.info("Starting scheduled detection run")
    start_time = datetime.utcnow()

    try:
        db = SessionLocal()
        try:
            result = run_detection_pipeline(db)
            duration = (datetime.utcnow() - start_time).total_seconds()
            logger.info(
                f"Scheduled detection completed in {duration:.1f}s. "
                f"Features: {result['features_created']}, "
                f"Tukey outliers: {result['tukey_outliers']}, "
                f"IF outliers: {result['isolation_forest_outliers']}, "
                f"Incidents created: {result['incidents_created']}, "
                f"Incidents updated: {result['incidents_updated']}"
            )
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Scheduled detection failed: {e}", exc_info=True)


def get_scheduler_status() -> dict:
    """Get current scheduler status."""
    global _scheduler

    if _scheduler is None:
        return {"running": False, "jobs": []}

    jobs = []
    for job in _scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
        })

    return {
        "running": _scheduler.running,
        "jobs": jobs,
    }


def trigger_detection_now():
    """Manually trigger a detection run."""
    global _scheduler

    if _scheduler is None:
        raise RuntimeError("Scheduler not initialised")

    _scheduler.add_job(
        _run_detection_job,
        id="manual_detection",
        name="Manual Detection Run",
        replace_existing=True,
    )
