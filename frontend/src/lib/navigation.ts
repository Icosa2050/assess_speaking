export const PRACTICE_PATHS: readonly string[] = [
  "/runtime-setup",
  "/session-setup",
  "/speak",
  "/rehearsal",
  "/review",
];

export const isPracticePath = (pathname: string): boolean => PRACTICE_PATHS.includes(pathname);
