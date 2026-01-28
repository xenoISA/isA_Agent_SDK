#!/bin/bash
# Agent Service Migration Runner
# Runs database migrations for trace system

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Database connection info (from post_info.md)
CONTAINER="staging-postgres"
DB_USER="postgres"
DB_NAME="isa_platform"

echo "========================================="
echo "Running Agent Service Migrations"
echo "========================================="
echo ""

# Run migrations in order
for migration in "$SCRIPT_DIR"/00*.sql; do
    if [ -f "$migration" ]; then
        filename=$(basename "$migration")
        echo "Running: $filename"
        docker exec -i "$CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" < "$migration"
        echo "✓ $filename completed"
        echo ""
    fi
done

echo "========================================="
echo "All migrations completed!"
echo "========================================="
echo ""
echo "Verify with:"
echo "  docker exec -i $CONTAINER psql -U $DB_USER -d $DB_NAME -c '\\dt agent.*'"
