#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Author:        Paul Calnon
# Version:       1.0.0
# File Name:     git_branch_ages.bash
# File Path:     <Project>/<Sub-Project>/<Application>/util/
#
# Date:          2025-12-03
# Last Modified: 2025-12-25
#
# License:       MIT License
# Copyright:     Copyright (c) 2024,2025,2026 Paul Calnon
#
# Description:
#     This script returns the ages of the current git branches.  Help to identify orphaned branches, etc.
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
set -o functrace
# # shellcheck disable=SC2155
# export PARENT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")" && INIT_CONF="../conf/init.conf"
# # shellcheck disable=SC1090,SC2015
# [[ -f "${INIT_CONF}" ]] && source "${INIT_CONF}" || { echo "Init Config File Not Found. Unable to Continue."; exit 1; }

# shellcheck disable=SC2155
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
export PARENT_PATH_PARAM="${SCRIPT_PATH}"
INIT_CONF="$(realpath "${SCRIPT_DIR}/../conf/init.conf")"
# echo "get_code_stats.bash: SCRIPT_PATH: ${SCRIPT_PATH}"
# echo "get_code_stats.bash: INIT_CONF: ${INIT_CONF}"
# shellcheck disable=SC2015,SC1091 source=conf/init.conf
[[ -f "${INIT_CONF}" ]] && source "${INIT_CONF}" || { echo "Init Config File Not Found at ${INIT_CONF}. Unable to Continue."; exit 1; }


#####################################################################################################################################################################################################
# Get git branches and ages.  yay.
#####################################################################################################################################################################################################
git fetch --prune

echo -ne "\nLocal Branches:\n"
git for-each-ref --sort='committerdate:iso8601' --color --format="%(color:green)%(committerdate:iso8601)|%(color:blue)%(committerdate:relative)|%(color:reset)%09%(refname)" refs/heads | awk -F "refs/heads/" '{print $1 $2;}' | column -s '|' -t

echo -ne "\nRemote Branches:\n"
git for-each-ref --sort='committerdate:iso8601' --color --format="%(color:green)%(committerdate:iso8601)|%(color:blue)%(committerdate:relative)|%(color:reset)%09%(refname)" refs/remotes | awk -F "refs/remotes/" '{print $1 $2;}' | column -s '|' -t

# [[ "${DEBUG}" == "${TRUE}" ]] && exit $(( TRUE )) || return $(( TRUE ))
# exit $(( TRUE ))
return $(( TRUE ))
