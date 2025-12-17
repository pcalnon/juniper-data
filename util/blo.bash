#!/usr/bin/env bash

export A="$(cat /proc/4128747/cmdline | tr -d '\0')"
echo "Command Line: \"${A}\""

echo "hex me"
cat /proc/4128747/cmdline | hd

echo "vanila b"
cat /proc/4128747/cmdline
