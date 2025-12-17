#!/usr/bin/env bash


export PARENT_SCRIPT_PATH_PARAM="$(realpath "${BASH_SOURCE[0]}")"
export INIT_CONF="../conf/init.conf"
source "${INIT_CONF}"; SUCCESS="$?"

[[ "${SUCCESS}" != "0" ]] && printf "%b%-21s %-28s %-21s %-11s %s%b\n" "\033[1;31m" "($(date +%F_%T))" "$(basename "${PARENT_SCRIPT_PATH_PARAM}"):(${LINENO})" "main:" "[CRITICAL]" "Config load Failed: \"${INIT_CONF}\"" "\033[0m" | tee -a "${LOG_FILE}" 2>&1 && set -e && exit 1
log_debug "Successfully Sourced Current Script: $(basename "${PARENT_SCRIPT_PATH_PARAM}"), Init Config File: ${INIT_CONF}, Success: ${SUCCESS}"


git fetch --prune

#for k in $(git branch | sed s/^..//); do echo -e $(git log --color=always -1 --pretty=format:"%Cgreen%ci %Cblue%cr%Creset" $k --)\\t"$k";done | sort
echo -ne "\nLocal Branches:\n"
git for-each-ref --sort='committerdate:iso8601' --color --format="%(color:green)%(committerdate:iso8601)|%(color:blue)%(committerdate:relative)|%(color:reset)%09%(refname)" refs/heads | awk -F "refs/heads/" '{print $1 $2;}' | column -s '|' -t

echo -ne "\nRemote Branches:\n"
git for-each-ref --sort='committerdate:iso8601' --color --format="%(color:green)%(committerdate:iso8601)|%(color:blue)%(committerdate:relative)|%(color:reset)%09%(refname)" refs/remotes | awk -F "refs/remotes/" '{print $1 $2;}' | column -s '|' -t
