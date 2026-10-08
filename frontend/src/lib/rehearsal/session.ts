import type { CefrLevel, TaskFamily } from "@/lib/state/sessionDraft";

export type RehearsalLanguage = "en" | "it";
export type RehearsalPart = {
  id: string; prompt: string; durationSec: number; taskFamily?: TaskFamily;
  recorded: boolean; audioId?: string; jobId?: string; reportId?: string;
  retryOf?: string; missingRecording?: boolean;
};
export type Rehearsal = {
  archivedAt?: string; restoredFrom?: string;
  version: 1; revision: number; id: string; createdAt: string; language: RehearsalLanguage;
  goal: CefrLevel; speaker: string; preparationSec: number; preparationEndsAt?: number;
  phase: "ready" | "preparation" | "speaking" | "review";
  parts: RehearsalPart[];
  runtime: { provider: string; model: string; baseUrl: string; whisper: string; feedbackLanguage: string };
};

const prompts = {
  en: {
    B1: ["Describe a memorable visit. Say where you went, what happened and why you remember it.", "Would you prefer to live in a city or a small town? Give reasons and an example.", "Plan a day out with a friend who has a small budget. Suggest two options, explain a difficulty and choose a plan."],
    B2: ["Describe a decision that changed your routine. Explain the alternatives, the outcome and what you learned.", "Should employers offer flexible working hours? Present your view, address an opposing argument and conclude.", "Your community wants a new meeting place but funding is limited. Compare two proposals, respond to concerns about cost and propose a compromise."],
    C1: ["Explain a complex decision you have observed. Weigh the competing interests and reflect on its longer-term consequences.", "Should public funding prioritise cultural heritage or new creative work? Build a nuanced argument, qualify your claims and address a counterargument.", "A city plans to restrict car access to its centre. Present a proposal that balances residents, local businesses and environmental concerns; anticipate objections and negotiate a feasible compromise."],
  },
  it: {
    B1: ["Racconta una visita che ricordi bene. Spiega dove sei andato, che cosa è successo e perché la ricordi.", "Preferiresti vivere in una città o in un piccolo paese? Esprimi la tua opinione con motivi e un esempio.", "Organizza una giornata con un amico che ha un budget limitato. Proponi due possibilità, spiega una difficoltà e scegli un programma."],
    B2: ["Racconta una decisione che ha cambiato le tue abitudini. Spiega le alternative, il risultato e che cosa hai imparato.", "Le aziende dovrebbero offrire orari di lavoro flessibili? Presenta la tua posizione, considera un'obiezione e concludi.", "Il tuo quartiere vuole un nuovo luogo d'incontro, ma i fondi sono limitati. Confronta due proposte, rispondi alle preoccupazioni sui costi e proponi un compromesso."],
    C1: ["Illustra una decisione complessa che hai osservato. Valuta gli interessi in contrasto e rifletti sulle conseguenze a lungo termine.", "I finanziamenti pubblici dovrebbero privilegiare il patrimonio culturale o le nuove opere creative? Sviluppa un'argomentazione articolata, precisa i limiti della tua posizione e considera un'obiezione.", "Una città intende limitare l'accesso delle auto al centro. Elabora una proposta che concili le esigenze dei residenti, dei commercianti e dell'ambiente; anticipa le obiezioni e negozia un compromesso realizzabile."],
  },
};

export function createRehearsal(language: RehearsalLanguage, goal: CefrLevel, speaker: string, runtime: Rehearsal["runtime"]): Rehearsal {
  return { version: 1, revision: 0, id: crypto.randomUUID(), createdAt: new Date().toISOString(), language, goal, speaker: speaker.trim(),
    preparationSec: 300, phase: "ready", runtime,
    parts: prompts[language][goal].map((prompt, index) => ({ id: `rehearsal_v1_${language}_${goal}_${index + 1}`, prompt,
      durationSec: [180, 180, 240][index], taskFamily: (["personal_experience", "opinion_monologue", "free_monologue"] as const)[index], recorded: false })) };
}

export function retryPart(session: Rehearsal, index: number): Rehearsal {
  const part = session.parts[index];
  if (!part?.reportId) throw new Error("A saved attempt is required for a linked retry.");
  return { ...session, revision: 0, id: crypto.randomUUID(), createdAt: new Date().toISOString(), preparationSec: 60,
    preparationEndsAt: undefined, phase: "ready",
    parts: [{ id: part.id, prompt: part.prompt, durationSec: part.durationSec, taskFamily: part.taskFamily, recorded: false, retryOf: part.reportId }] };
}

export function secondsRemaining(deadline: number, now = Date.now()): number {
  return Math.max(0, Math.ceil((deadline - now) / 1000));
}
export const clockText = (seconds: number) => `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
