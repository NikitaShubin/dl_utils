import type { Plugin } from "@opencode-ai/plugin"
import { closeSync, openSync, writeSync } from "node:fs"

const NEEDS_USER = new Set([
  "session.idle",
  "permission.asked",
  "permission.v2.asked",
  "question.asked",
  "question.v2.asked",
  "session.error",
])

export const BellPlugin: Plugin = async () => ({
  event: async ({ event }) => {
    if (!NEEDS_USER.has(event.type)) return
    try {
      const fd = openSync("/dev/tty", "w")
      writeSync(fd, "\x07")
      closeSync(fd)
    } catch {}
  },
})
