#!/bin/sh
# Create the Oracle SQLite database from /schema/schema.sql, or report that it already exists.
# Idempotent: safe to run any number of times. Never modifies an existing database.
set -eu

db="${ORACLE_DB_FILE:-/data/oracle.sqlite3}"
schema="${ORACLE_SCHEMA_FILE:-/schema/schema.sql}"

if [ ! -f "$schema" ]; then
    echo "This script runs inside the db-init container. From the repo root use one of:" >&2
    echo "  docker compose run --rm db-init" >&2
    echo "  uv run --env-file .env oracle status   (any oracle command creates ./data/oracle.sqlite3)" >&2
    exit 1
fi

version=$(sqlite3 "$db" 'PRAGMA user_version;')
current=$(sed -n 's/^PRAGMA user_version = \([0-9][0-9]*\);.*/\1/p' "$schema")
if [ "$version" -eq 0 ]; then
    sqlite3 -bail "$db" < "$schema"
    sqlite3 "$db" 'PRAGMA journal_mode = WAL;' > /dev/null
    echo "Created $db at schema v$(sqlite3 "$db" 'PRAGMA user_version;')"
elif [ "$version" -lt "$current" ]; then
    echo "$db is at schema v$version, current is v$current. Not touched: any \`oracle\` command (for example \`oracle status\`) upgrades it in place."
else
    echo "$db already exists at schema v$version; nothing to do"
fi
