#!/usr/bin/env bash

# This script's only job is to execute the Python handler.
# 'exec' ensures the Python process replaces the shell process, which is good practice.
exec python /handler.py
