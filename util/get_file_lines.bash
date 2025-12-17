#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Script Name:   get_file_lines.bash
# Script Path:   <Project>/<Sub-Project>/juniper_canopy/util/get_file_lines.bash
# Conf File:     get_file_lines.conf
# Conf Path:     <Project>/<Sub-Project>/<Application>/conf/  # TODO: Add parent project dir
#
# Author:        Paul Calnon
# Version:       1.0.0
#
# Date:          2025-12-03
# Last Modified: 2025-12-15
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     This script is used to get the number of lines in a file.
#
#####################################################################################################################################################################################################
# Notes:
#
#####################################################################################################################################################################################################
# References:
#
#####################################################################################################################################################################################################
# TODO :
#
#####################################################################################################################################################################################################
# COMPLETED:
#
#####################################################################################################################################################################################################


#####################################################################################################################################################################################################
# Source script config file
#####################################################################################################################################################################################################
# set -eE -o functrace
export PARENT_SCRIPT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")"
export INIT_CONF="../conf/init.conf"
source "${INIT_CONF}"; SUCCESS="$?"

[[ "${SUCCESS}" != "0" ]] && printf "%b%-21s %-28s %-21s %-11s %s%b\n" "\033[1;31m" "($(date +%F_%T))" "$(basename "${PARENT_SCRIPT_PATH_PARAM}"):(${LINENO})" "main:" "[CRITICAL]" "Config load Failed: \"${INIT_CONF}\"" "\033[0m" | tee -a "${LOG_FILE}" 2>&1 && set -e && exit 1
log_debug "Successfully Sourced Current Script: $(basename "${PARENT_SCRIPT_PATH_PARAM}"), Init Config File: ${INIT_CONF}, Success: ${SUCCESS}"

# source "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")/conf/$(basename -s ".bash" "$(realpath "${BASH_SOURCE[0]}")").conf"; SUCCESS="$?"
# [[ "${SUCCESS}" != "0" ]] && printf "%b%-21s %-28s %-21s %-11s %s%b\n" "\033[1;31m" "($(date +%F_%T))" "$(basename "${SCRIPT_PATH}"):(${LINENO})" "main:" "[CRITICAL]" "Config load Failed: \"${CONF_FILE}\"" "\033[0m" | tee -a "${LOG_FILE}" 2>&1 && set -e && exit 1
# log_debug "Successfully Sourced Current Script: ${SCRIPT_NAME}, Config File: ${CONF_FILE}, Success: ${SUCCESS}"


################################################################################################################
# TODO: Use this to get names & then get lines
################################################################################################################
TOTAL_FILES=0
TOTAL_LINES=0
TOTAL_TODOS=0

for i in $(${GET_FILENAMES_SCRIPT_NAME}); do

  TOTAL_FILES=$(( TOTAL_FILES + 1 ))
  CURRENT_LINES="$(cat ${i} | wc -l)"
  TOTAL_LINES=$(( TOTAL_LINES + CURRENT_LINES ))
  CURRENT_TODOS="$(${TODO_SEARCH_SCRIPT} ${i})"
  TOTAL_TODOS=$(( TOTAL_TODOS + CURRENT_TODOS ))

  CURRENT_SIZE="$(du -sh ${i})"

  echo "File: ${i}\tLines: ${CURRENT_LINES}\tTODOs: ${CURRENT_TODOS}"

done


#######################################################################################################################################################################################
# Display Results Summary
#######################################################################################################################################################################################
echo "Search Term: \"${SEARCH_TERM}\""
echo "Found in Files: ${FOUND_COUNT}"
echo "Files Complete: ${DONE_COUNT}"
