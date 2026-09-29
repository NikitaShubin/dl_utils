import type { Plugin } from "@opencode/plugin"
import { closeSync, openSync, writeSync } from "node:fs"

// BEL-звонок в терминал, когда OpenCode ждёт пользователя.
// API плагинов v2: https://opencode.ai/v2/docs/build/plugins
// Импорт @opencode/plugin нужен только для типов — во время исполнения он
// стирается, поэтому плагин работает без установки пакетов в node_modules.

// Звонок сразу: ход сессии завершён (успешно или с ошибкой):
const RING_NOW = new Set(["session.execution.succeeded", "session.execution.failed"])

// Звонок с задержкой: запрашивают ответа пользователя, но если он успел
// ответить раньше, звонок отменяется:
const ASKED = new Set(["permission.asked", "form.created"])
const RESOLVED = new Set(["permission.replied", "form.replied", "form.cancelled"])

const GRACE_MS = 300

function ring(): void {
  try {
    const fd = openSync("/dev/tty", "w")
    writeSync(fd, "\x07")
    closeSync(fd)
  } catch {}
}

export default {
  id: "bell",
  setup(ctx) {
    const controller = new AbortController()
    const pending = new Map<string, ReturnType<typeof setTimeout>>()

    void (async () => {
      for await (const event of ctx.event.subscribe({ signal: controller.signal })) {
        if (RING_NOW.has(event.type)) {
          ring()
          continue
        }

        // Идентификатор запроса лежит в разных полях разных событий:
        let id: string | undefined
        switch (event.type) {
          case "permission.asked":
            id = event.data.id
            break
          case "form.created":
            id = event.data.form.id
            break
          case "permission.replied":
            id = event.data.requestID
            break
          case "form.replied":
          case "form.cancelled":
            id = event.data.id
            break
        }
        if (id === undefined) continue

        if (ASKED.has(event.type)) {
          if (pending.has(id)) continue
          pending.set(
            id,
            setTimeout(() => {
              pending.delete(id)
              ring()
            }, GRACE_MS),
          )
        } else if (RESOLVED.has(event.type)) {
          const timer = pending.get(id)
          if (timer) {
            clearTimeout(timer)
            pending.delete(id)
          }
        }
      }
    })()

    // Вызывается при выгрузке плагина: останавливаем подписку и таймеры:
    return () => {
      controller.abort()
      for (const timer of pending.values()) clearTimeout(timer)
      pending.clear()
    }
  },
} satisfies Plugin
