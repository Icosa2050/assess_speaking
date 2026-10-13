import { type BrowserContext, test as base, expect } from "@playwright/test";
/** Browser traffic is fixture-only as well as backend inference traffic. */
export const test = base.extend({
  context: async ({ context }, use) => {
    const verify = await guardBrowserContext(context);
    await use(context);
    verify();
  },
});
export { expect };
export type { Page, APIRequestContext, TestInfo, Route } from "@playwright/test";

export async function guardBrowserContext(context: BrowserContext) {
    const blocked: string[] = [];
    await context.route("**/*", async route => {
      const url = new URL(route.request().url());
      if (!["http:", "https:"].includes(url.protocol) || ["127.0.0.1", "localhost", "[::1]"].includes(url.hostname)) return route.continue();
      blocked.push(`${url.protocol}//${url.hostname}`);
      await route.abort("blockedbyclient");
    });
  return () => expect(blocked, "Unexpected external browser requests").toEqual([]);
}
