#!/usr/bin/env bash
# launchd işlerini kurar: 23:30 nightly (caffeinate ile uyku engeli) ve 09:15 morning.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# janus çalıştırılabilir dizini; ortamınıza göre JANUS_BIN ile geçersiz kılın.
JANUS_BIN="${JANUS_BIN:-$ROOT/.venv/bin}"
mkdir -p "$HOME/Library/Logs/janus" "$HOME/Library/LaunchAgents"
for job in nightly morning; do
  src="$ROOT/scripts/launchd/com.janus.$job.plist"
  dst="$HOME/Library/LaunchAgents/com.janus.$job.plist"
  sed -e "s#__JANUS_ROOT__#$ROOT#g" -e "s#__HOME__#$HOME#g" -e "s#__JANUS_BIN__#$JANUS_BIN#g" "$src" > "$dst"
  launchctl unload "$dst" 2>/dev/null || true
  launchctl load "$dst"
  echo "yüklendi: $dst"
done
echo "Not: Mac'in gece uyanık kalması için Sistem Ayarları > Enerji > 'Ağ erişimi için uyandır' açık olmalı; caffeinate koşu süresince uykuyu engeller."
