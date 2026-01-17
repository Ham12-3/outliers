"""Add flexible data model for any CSV structure.

Revision ID: 002
Revises: 001
Create Date: 2025-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '002'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create datasets table to track uploaded datasets and their schema
    op.create_table(
        'datasets',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),

        # Schema configuration (user-defined)
        sa.Column('date_column', sa.String(100), nullable=True),
        sa.Column('identifier_columns', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('metric_columns', sa.JSON(), nullable=False, server_default='[]'),
        sa.Column('attribute_columns', sa.JSON(), nullable=False, server_default='[]'),

        # Column analysis (auto-detected)
        sa.Column('column_analysis', sa.JSON(), nullable=True),

        # Stats
        sa.Column('row_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('date_range_start', sa.Date(), nullable=True),
        sa.Column('date_range_end', sa.Date(), nullable=True),

        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),

        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_datasets_name', 'datasets', ['name'])

    # Create data_rows table for flexible data storage
    op.create_table(
        'data_rows',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('dataset_id', sa.Integer(), nullable=False),

        # Date (parsed from user-specified column)
        sa.Column('date', sa.Date(), nullable=True),

        # Identifier values as JSON (e.g., {"store_id": "LONDON-001", "product_id": "SKU-123"})
        sa.Column('identifiers', sa.JSON(), nullable=False, server_default='{}'),

        # Metric values as JSON (e.g., {"sales": 100, "inventory": 500})
        sa.Column('metrics', sa.JSON(), nullable=False, server_default='{}'),

        # Attribute values as JSON (e.g., {"promo": true, "category": "Food"})
        sa.Column('attributes', sa.JSON(), nullable=False, server_default='{}'),

        # Computed identifier key for grouping (hash of identifier values)
        sa.Column('identifier_key', sa.String(64), nullable=True),

        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),

        sa.ForeignKeyConstraint(['dataset_id'], ['datasets.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_data_rows_dataset', 'data_rows', ['dataset_id'])
    op.create_index('ix_data_rows_date', 'data_rows', ['date'])
    op.create_index('ix_data_rows_identifier_key', 'data_rows', ['identifier_key'])
    op.create_index('ix_data_rows_dataset_date', 'data_rows', ['dataset_id', 'date'])
    op.create_index('ix_data_rows_dataset_identifier', 'data_rows', ['dataset_id', 'identifier_key'])

    # Update detection_results to reference datasets
    op.add_column('detection_results', sa.Column('dataset_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_detection_results_dataset',
        'detection_results', 'datasets',
        ['dataset_id'], ['id'],
        ondelete='CASCADE'
    )

    # Add metric_name flexibility (can now be any user-defined metric)
    # Already exists, just updating for flexibility

    # Update incidents to reference datasets
    op.add_column('incidents', sa.Column('dataset_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_incidents_dataset',
        'incidents', 'datasets',
        ['dataset_id'], ['id'],
        ondelete='CASCADE'
    )


def downgrade() -> None:
    # Remove foreign keys
    op.drop_constraint('fk_incidents_dataset', 'incidents', type_='foreignkey')
    op.drop_column('incidents', 'dataset_id')

    op.drop_constraint('fk_detection_results_dataset', 'detection_results', type_='foreignkey')
    op.drop_column('detection_results', 'dataset_id')

    # Drop new tables
    op.drop_table('data_rows')
    op.drop_table('datasets')
