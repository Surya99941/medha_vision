import sys
from os.path import abspath, dirname
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

# Add the project root to the Python path
sys.path.insert(0, abspath(dirname(dirname(dirname(__file__)))))
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from app.models import Base
# target_metadata = Base.metadata
target_metadata = None

# ... (rest of the file is the same until run_migrations_online)
# ...
def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    # Get the database URL from the app's config
    from app.database import DATABASE_URL, Base
    target_metadata = Base.metadata
    config.set_main_option('sqlalchemy.url', DATABASE_URL)

    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
