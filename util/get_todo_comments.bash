#!/usr/bin/env bash
#####################################################################################################################################################################################################
# Project:       Juniper
# Sub-Project:   JuniperCanopy
# Application:   juniper_canopy
# Purpose:       Monitoring and Diagnostic Frontend for Cascade Correlation Neural Network
#
# Script Name:   get_todo_comments.bash
# Script Path:   <Project>/<Sub-Project>/juniper_canopy/util/get_todo_comments.bash
# Conf File:     get_todo_comments.conf
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
#     This script collects Descriptions associated with all TODO Comments in the Juniper python project code base
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

# source "$(dirname "$(dirname "$(realpath "${BASH_SOURCE[0]}")")")/conf/$(basename -s ".bash" "$(realpath "${BASH_SOURCE[0]}")").conf"; SUCCESS="$?"
# [[ "${SUCCESS}" != "0" ]] && printf "%b%-21s %-28s %-21s %-11s %s%b\n" "\033[1;31m" "($(date +%F_%T))" "$(basename "${SCRIPT_PATH}"):(${LINENO})" "main:" "[CRITICAL]" "Config load Failed: \"${CONF_FILE}\"" "\033[0m" | tee -a "${LOG_FILE}" 2>&1 && set -e && exit 1
# log_debug "Successfully Sourced Current Script: ${SCRIPT_NAME}, Config File: ${CONF_FILE}, Success: ${SUCCESS}"

export PARENT_SCRIPT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")"
export INIT_CONF="../conf/init.conf"
source "${INIT_CONF}"; SUCCESS="$?"

[[ "${SUCCESS}" != "0" ]] && printf "%b%-21s %-28s %-21s %-11s %s%b\n" "\033[1;31m" "($(date +%F_%T))" "$(basename "${PARENT_SCRIPT_PATH_PARAM}"):(${LINENO})" "main:" "[CRITICAL]" "Config load Failed: \"${INIT_CONF}\"" "\033[0m" | tee -a "${LOG_FILE}" 2>&1 && set -e && exit 1
log_debug "Successfully Sourced Current Script: $(basename "${PARENT_SCRIPT_PATH_PARAM}"), Init Config File: ${INIT_CONF}, Success: ${SUCCESS}"




#####################################################################################################
# Define Global Configuration File Constants
####################################################################################################
# ROOT_PROJ_NAME="dynamic_nn"
# ROOT_PROJ_NAME="juniper"
ROOT_PROJ_NAME="Juniper"
ROOT_CONF_NAME="conf"
ROOT_CONF_FILE_NAME="common.conf"
#ROOT_PROJ_DIR="${HOME}/Development/rust/rust_mudgeon/${ROOT_PROJ_NAME}"
ROOT_PROJ_DIR="${HOME}/Development/python/${ROOT_PROJ_NAME}"
ROOT_CONF_DIR="${ROOT_PROJ_DIR}/${ROOT_CONF_NAME}"
ROOT_CONF_FILE="${ROOT_CONF_DIR}/${ROOT_CONF_FILE_NAME}"
INIT_PYTHON_FILE="__init__"
source ${ROOT_CONF_FILE}
#echo "Root Conf File: ${ROOT_CONF_FILE}"
#echo "Source Dir Name: ${SOURCE_DIR_NAME}"


##################################################################################
# Determine Project Dir
##################################################################################
#BASE_DIR=$(${GET_PROJECT_SCRIPT} "${BASH_SOURCE}")
BASE_DIR=${ROOT_PROJ_DIR}
#echo "Base Dir: ${BASE_DIR}"


##################################################################################
# Determine Host OS
##################################################################################
CURRENT_OS=$(${GET_OS_SCRIPT})
#echo "Current OS: ${CURRENT_OS}"


####################################################################################################
# Define Script Functions
####################################################################################################
source ${DATE_FUNCTIONS_SCRIPT}

function usage() {
    EXIT_COND="${EXIT_COND_DEFAULT}"
    if [[ "$1" != "" ]]; then
        EXIT_COND="$1"
    fi
    MESSAGE="usage: ${SCRIPT_NAME} <SEARCH TERM> | [--help|-h]"
    echo -ne "\n\t${MESSAGE}\n\n"
    exit ${EXIT_COND}
}


#######################################################################################################################################################################################
# Define the Script Environment Constants
#######################################################################################################################################################################################
SRC_DIR="${BASE_DIR}/${SOURCE_DIR_NAME}"
#echo "Source Dir: ${SRC_DIR}"
LOG_DIR="${BASE_DIR}/${LOGGING_DIR_NAME}"
#echo "Log Dir: ${LOG_DIR}"
CONF_DIR="${BASE_DIR}/${CONFIG_DIR_NAME}"
#echo "Conf Dir: ${CONF_DIR}"

##LOG_FILE_MODULE="${LOG_DIR}/${LOG_FILE_MODULE_NAME}"
##echo "Log File Module: ${LOG_FILE_MODULE}"
##LOG_FILE_STARTUP="${LOG_DIR}/${LOG_FILE_STARTUP_NAME}"
##echo "Log File Startup: ${LOG_FILE_STARTUP}"
CONF_FILE="${CONF_DIR}/${ROOT_CONF_FILE_NAME}"
#echo "Conf File: ${CONF_FILE}"


#######################################################################################################################################################################################
# Define the Script Constants
#######################################################################################################################################################################################
#DEBUG="true"
DEBUG="false"

FULL_OUTPUT="true"
#FULL_OUTPUT="false"

HELP_SHORT="-h"
HELP_LONG="--help"

EXIT_COND_DEFAULT="1"

#SEARCH_TERM_DEFAULT="write tests"
SEARCH_TERM_DEFAULT="TODO"


#######################################################################################################################################################################################
# Process Script's Command Line Argument(s)
#######################################################################################################################################################################################
if [[ "$1" != "" ]]; then
    if [[ "$1" == "${HELP_SHORT}" || "$1" == "${HELP_LONG}" ]]; then
        usage 0
    else
        SEARCH_TERM="$1"
    fi
else
    #SEARCH_TERM="${SEARCH_TERM_DEFAULT}"
    if [[ ${DEBUG} == "true" ]]; then
        SEARCH_TERM="${SEARCH_TERM_DEFAULT}"
    else
        usage
    fi
fi


#######################################################################################################################################################################################
# Sanitize Inputs
#######################################################################################################################################################################################
DASHES=$(echo "${SEARCH_TERM}" | grep -e '^-.*')
if [[ ${DASHES} != "" ]]; then
    SEARCH_TERM="\\${SEARCH_TERM}"
    if [[ ${DEBUG} == "true" ]]; then
        echo "Sanitized SEARCH_TERM Input: ${SEARCH_TERM}"
    fi
fi


#######################################################################################################################################################################################
# Search for a specific TODO reference in source code
#######################################################################################################################################################################################
DONE_COUNT=0
FOUND_COUNT=0
for i in $(find ${SRC_DIR}); do
    SOURCE_FILE=$(echo "${i}" | grep "\.${SRC_FILE_SUFFIX}\$")
    #echo "Source File: \"${SOURCE_FILE}\""
    if [[ ${SOURCE_FILE} != "" ]]; then
        SOURCE_FILE=$(echo "${SOURCE_FILE}" | grep -v "${INIT_PYTHON_FILE}")
	if [[ ${SOURCE_FILE} != "" ]]; then
            if [[ -f ${SOURCE_FILE} ]]; then
                FOUND=$(cat ${SOURCE_FILE} | grep "${SEARCH_TERM}")
                if [[ ${FOUND} != "" ]]; then
                    FOUND_COUNT=$((FOUND_COUNT + 1))
                    if [[ ${DEBUG} == "true" || ${FULL_OUTPUT} == "true" ]]; then
                        #echo -ne "${SOURCE_FILE}\n\tFound: ${FOUND}\n\n"
                        echo -ne "${SOURCE_FILE}\n${FOUND}\n\n"
                    fi
                else
                    DONE_COUNT=$((DONE_COUNT + 1))
                    if [[ ${FULL_OUTPUT} == "true" && ${DEBUG} == "true" ]]; then
                        echo -ne "${SOURCE_FILE}\n\tNot Found: **********************\n\n"
                    fi
                fi
            fi
        fi
    fi
done


#######################################################################################################################################################################################
# Display Results
#######################################################################################################################################################################################
echo "Search Term: \"${SEARCH_TERM}\""
echo "Found in Files: ${FOUND_COUNT}"
echo "Files Complete: ${DONE_COUNT}"
