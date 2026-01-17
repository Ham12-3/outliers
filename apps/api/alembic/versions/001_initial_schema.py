"""Initial schema for inventory anomaly detection.

Revision ID: 001
Revises:
Create Date: 2025-01-17

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = '001'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Define enums with create_type=False so we control creation with checkfirst
incidentstatus_enum = postgresql.ENUM(
    'open',
    'investigating',
    'resolved',
    name='incidentstatus',
    create_type=False,
)

detectortype_enum = postgresql.ENUM(
    'tukey',
    'isolation_forest',
    name='detectortype',
    create_type=False,
)


def upgrade() -> None:
    # Create enum types safely (checkfirst=True prevents error if they exist)
    bind = op.get_bind()
    incidentstatus_enum.create(bind, checkfirst=True)
    detectortype_enum.create(bind, checkfirst=True)

    # raw_daily_metrics table
    op.create_table(
        'raw_daily_metrics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('store_id', sa.String(50), nullable=False),
        sa.Column('sku_id', sa.String(50), nullable=False),
        sa.Column('on_hand', sa.Integer(), nullable=False),
        sa.Column('sold', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('delivered', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('returned', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('promo_flag', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date', 'store_id', 'sku_id', name='uq_daily_metric')
    )
    op.create_index('ix_raw_daily_metrics_date_store_sku', 'raw_daily_metrics', ['date', 'store_id', 'sku_id'])
    op.create_index('ix_raw_daily_metrics_store', 'raw_daily_metrics', ['store_id'])
    op.create_index('ix_raw_daily_metrics_sku', 'raw_daily_metrics', ['sku_id'])
    op.create_index('ix_raw_daily_metrics_date', 'raw_daily_metrics', ['date'])

    # features_daily table
    op.create_table(
        'features_daily',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('store_id', sa.String(50), nullable=False),
        sa.Column('sku_id', sa.String(50), nullable=False),
        sa.Column('on_hand', sa.Integer(), nullable=False),
        sa.Column('sold', sa.Integer(), nullable=False),
        sa.Column('delivered', sa.Integer(), nullable=False),
        sa.Column('returned', sa.Integer(), nullable=False),
        sa.Column('price', sa.Float(), nullable=False),
        sa.Column('promo_flag', sa.Boolean(), nullable=False),
        sa.Column('delta_on_hand', sa.Integer(), nullable=True),
        sa.Column('day_of_week', sa.Integer(), nullable=False),
        sa.Column('sold_rolling_mean', sa.Float(), nullable=True),
        sa.Column('sold_rolling_std', sa.Float(), nullable=True),
        sa.Column('on_hand_rolling_mean', sa.Float(), nullable=True),
        sa.Column('on_hand_rolling_std', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('date', 'store_id', 'sku_id', name='uq_feature_daily')
    )
    op.create_index('ix_features_daily_date_store_sku', 'features_daily', ['date', 'store_id', 'sku_id'])
    op.create_index('ix_features_daily_date', 'features_daily', ['date'])

    # detection_results table
    op.create_table(
        'detection_results',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('store_id', sa.String(50), nullable=False),
        sa.Column('sku_id', sa.String(50), nullable=False),
        sa.Column('detector_type', detectortype_enum, nullable=False),
        sa.Column('metric_name', sa.String(50), nullable=True),
        sa.Column('q1', sa.Float(), nullable=True),
        sa.Column('q3', sa.Float(), nullable=True),
        sa.Column('iqr', sa.Float(), nullable=True),
        sa.Column('lower_fence', sa.Float(), nullable=True),
        sa.Column('upper_fence', sa.Float(), nullable=True),
        sa.Column('actual_value', sa.Float(), nullable=True),
        sa.Column('outlier_distance', sa.Float(), nullable=True),
        sa.Column('anomaly_score', sa.Float(), nullable=True),
        sa.Column('threshold_used', sa.Float(), nullable=True),
        sa.Column('is_outlier', sa.Boolean(), nullable=False, server_default='false'),
        sa.Column('reasons', sa.JSON(), nullable=True),
        sa.Column('sample_size', sa.Integer(), nullable=True),
        sa.Column('fallback_used', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_detection_results_date_store_sku', 'detection_results', ['date', 'store_id', 'sku_id'])
    op.create_index('ix_detection_results_is_outlier', 'detection_results', ['is_outlier'])
    op.create_index('ix_detection_results_detector', 'detection_results', ['detector_type'])

    # incidents table
    op.create_table(
        'incidents',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('date', sa.Date(), nullable=False),
        sa.Column('store_id', sa.String(50), nullable=False),
        sa.Column('status', incidentstatus_enum, nullable=False, server_default='open'),
        sa.Column('severity_score', sa.Float(), nullable=False),
        sa.Column('headline', sa.String(500), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('sku_count', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('estimated_impact', sa.Float(), nullable=True),
        sa.Column('detectors_triggered', sa.JSON(), nullable=False),
        sa.Column('assignee', sa.String(100), nullable=True),
        sa.Column('resolution_reason', sa.Text(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('dedup_key', sa.String(255), nullable=False),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('dedup_key')
    )
    op.create_index('ix_incidents_date', 'incidents', ['date'])
    op.create_index('ix_incidents_store', 'incidents', ['store_id'])
    op.create_index('ix_incidents_status', 'incidents', ['status'])
    op.create_index('ix_incidents_severity', 'incidents', ['severity_score'])
    op.create_index('ix_incidents_date_store', 'incidents', ['date', 'store_id'])

    # incident_items table
    op.create_table(
        'incident_items',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('incident_id', sa.Integer(), nullable=False),
        sa.Column('sku_id', sa.String(50), nullable=False),
        sa.Column('detection_result_ids', sa.JSON(), nullable=False),
        sa.Column('contribution_score', sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_incident_items_incident', 'incident_items', ['incident_id'])
    op.create_index('ix_incident_items_sku', 'incident_items', ['sku_id'])

    # incident_notes table
    op.create_table(
        'incident_notes',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('incident_id', sa.Integer(), nullable=False),
        sa.Column('author', sa.String(100), nullable=False, server_default='system'),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('note_type', sa.String(50), nullable=False, server_default='comment'),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['incident_id'], ['incidents.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_incident_notes_incident', 'incident_notes', ['incident_id'])


def downgrade() -> None:
    op.drop_table('incident_notes')
    op.drop_table('incident_items')
    op.drop_table('incidents')
    op.drop_table('detection_results')
    op.drop_table('features_daily')
    op.drop_table('raw_daily_metrics')

    # Drop enum types safely (checkfirst=True prevents error if they don't exist)
    bind = op.get_bind()
    incidentstatus_enum.drop(bind, checkfirst=True)
    detectortype_enum.drop(bind, checkfirst=True)
