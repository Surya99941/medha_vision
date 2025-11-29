"""add summary to image model

Revision ID: ff302390454c
Revises: 1394d18f3039
Create Date: 2025-11-29 10:33:27.691233

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ff302390454c'
down_revision: Union[str, Sequence[str], None] = '1394d18f3039'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('images', sa.Column('summary', sa.String(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('images', 'summary')
