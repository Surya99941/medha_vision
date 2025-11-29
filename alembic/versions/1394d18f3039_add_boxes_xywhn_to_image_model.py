"""Add boxes_xywhn to Image model

Revision ID: 1394d18f3039
Revises: a8094db41a93
Create Date: 2025-11-29 01:08:00.246637

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '1394d18f3039'
down_revision: Union[str, Sequence[str], None] = 'a8094db41a93'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('images', sa.Column('boxes_xywhn', sa.JSON(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('images', 'boxes_xywhn')
    pass
