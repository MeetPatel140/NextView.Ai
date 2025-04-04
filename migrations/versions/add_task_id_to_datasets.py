"""add task_id to datasets

Revision ID: add_task_id_to_datasets
Revises: 779471655e09
Create Date: 2025-03-27 00:24:31.769799

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'add_task_id_to_datasets'
down_revision = '779471655e09'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('datasets', sa.Column('task_id', sa.String(length=255), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('datasets', 'task_id')
    # ### end Alembic commands ###