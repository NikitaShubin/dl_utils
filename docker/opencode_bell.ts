import type { Plugin } from "@opencode-ai/plugin"
import { closeSync, openSync, writeSync } from "node:fs"

const RING_NOW = new Set(["session.idle", "session.error"])

const ASKED = new Set([
  "permission.asked",
  "permission.v2.asked",
  "question.asked",
  "question.v2.asked",
])

const RESOLVED = new Set([
  "permission.replied",
  "permission.v2.replied",
  "question.replied",
  "question.rejected",
])

const GRACE_MS = 300

const pending = new Map<string, ReturnType<typeof setTimeout>>()
let anonymous = 0

function requestId(props: unknown): string | undefined {
  if (!props || typeof props !== "object") return undefined
  const rec = props as Record<string, unknown>
  for (const key of ["id", "requestID", "permissionID"]) {
    const value = rec[key]
    if (typeof value === "string" && value) return value
  }
  return undefined
}

function bell(): void {
  try {
    const fd = openSync("/dev/tty", "w")
    writeSync(fd, "\x07")
    closeSync(fd)
  } catch {}
}

export const BellPlugin: Plugin = async () => ({
  event: async ({ event }) => {
    const props = "properties" in event ? event.properties : undefined

    if (RING_NOW.has(event.type)) {
      bell()
      return
    }

    if (ASKED.has(event.type)) {
      const id = requestId(props)
      const key = id ?? `#anon:${++anonymous}`
      if (id && pending.has(id)) return
      const timer = setTimeout(() => {
        pending.delete(key)
        bell()
      }, GRACE_MS)
      pending.set(key, timer)
      return
    }

    if (RESOLVED.has(event.type)) {
      const id = requestId(props)
      if (!id) return
      const timer = pending.get(id)
      if (timer) {
        clearTimeout(timer)
        pending.delete(id)
      }
    }
  },
})
