const FIRST_TAKE_KEY = "assess-speaking.first-take-recorded";

export const hasFirstTakeMilestone = (): boolean => {
  try {
    return window.localStorage.getItem(FIRST_TAKE_KEY) === "1";
  } catch {
    return false;
  }
};

export const recordFirstTakeMilestone = (): void => {
  try {
    window.localStorage.setItem(FIRST_TAKE_KEY, "1");
  } catch {
    // Recording remains usable when local storage is unavailable.
  }
};
