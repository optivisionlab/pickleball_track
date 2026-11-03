#!/usr/bin/env bash
# Keeps the scoring stack up: if fewer than AI_REPLICAS model servers or no nginx
# load balancer are running (stopped, removed or crashed), runs `docker compose up -d`.
# `restart: unless-stopped` alone does not bring back a container someone stopped.
#
# Install (every minute, from the user's crontab):
#   * * * * * /path/to/score_video_block/deploy/watchdog.sh >> $HOME/pickleball-ai-watchdog.log 2>&1
# Planned maintenance: `touch deploy/.watchdog-pause` to pause it, `rm` it to resume.
set -u
export PATH=/usr/local/bin:/usr/bin:/bin

DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR" || exit 1
[ -e deploy/.watchdog-pause ] && exit 0

# One run at a time (a start can take longer than the cron interval).
exec 9>"/tmp/pickleball-ai-watchdog-$(id -u).lock"
flock -n 9 || exit 0

# Same replica count as docker compose: environment, then .env, then 4.
want="${AI_REPLICAS:-$(sed -n 's/^AI_REPLICAS=//p' .env 2>/dev/null | tail -1)}"
want="${want:-4}"

ai=$(docker compose ps -q --status running pickleball-ai | wc -l)
lb=$(docker compose ps -q --status running pickleball-lb | wc -l)

if [ "$ai" -lt "$want" ] || [ "$lb" -lt 1 ]; then
    echo "$(date -Is) running ai=$ai/$want lb=$lb -> docker compose up -d"
    docker compose up -d --no-build
fi
