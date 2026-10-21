#!/bin/sh
# Create the Oracle SQLite database from /schema/schema.sql, or report that it already exists.
# Idempotent: safe to run any number of times. Never modifies an existing database.
set -eu

db="${ORACLE_DB_FILE:-/data/oracle.sqlite3}"

version=$(sqlite3 "$db" 'PRAGMA user_version;')
if [ "$version" -eq 0 ]; then
    sqlite3 -bail "$db" < /schema/schema.sql
    sqlite3 "$db" 'PRAGMA journal_mode = WAL;' > /dev/null
    echo "Created $db at schema v$(sqlite3 "$db" 'PRAGMA user_version;')"
else
    echo "$db already exists at schema v$version; nothing to do"
fi
