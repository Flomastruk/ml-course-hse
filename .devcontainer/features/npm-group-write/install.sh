#!/usr/bin/env bash
set -e
find "$(npm root -g)" -type d -exec chmod g+w {} +