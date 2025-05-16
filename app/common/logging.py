import logging


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    # ✅ Logs raw SQL queries + bound params
    logging.getLogger("sqlalchemy.engine").setLevel(logging.INFO)
