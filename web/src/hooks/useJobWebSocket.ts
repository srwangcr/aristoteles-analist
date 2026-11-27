import { useEffect, useRef, useState } from 'react';
import { connectJobSocket } from '../api';
import type { JobSocketHandlers, JobStatus, JobViewState } from '../types';

const INITIAL: JobViewState = {
  status: 'idle',
  progress: 0,
  message: '',
  socketConnected: false,
};

/**
 * Drives detection progress from the job WebSocket, with REST/reconnect
 * handling delegated to connectJobSocket(). Callbacks are held in refs so they
 * do not force a new socket subscription on every render.
 */
export function useJobWebSocket(
  jobId: string | null,
  handlers: JobSocketHandlers,
): JobViewState {
  const [state, setState] = useState<JobViewState>(INITIAL);
  const handlersRef = useRef(handlers);
  handlersRef.current = handlers;

  useEffect(() => {
    if (!jobId) {
      setState(INITIAL);
      return;
    }

    setState({ ...INITIAL, status: 'queued', message: 'Connecting to job stream…' });

    const dispose = connectJobSocket(jobId, {
      onProgress: (value, message) => {
        setState((previous) => ({
          ...previous,
          status: 'running',
          progress: Math.min(100, Math.max(0, value)),
          message,
          error: undefined,
        }));
        handlersRef.current.onProgress?.(value, message);
      },
      onDone: (elapsedS) => {
        setState((previous) => ({
          ...previous,
          status: 'done',
          progress: 100,
          message: previous.message || 'Detection complete',
          elapsedS,
          error: undefined,
        }));
        handlersRef.current.onDone?.(elapsedS);
      },
      onError: (message) => {
        setState((previous) => ({
          ...previous,
          status: 'error',
          message,
          error: message,
        }));
        handlersRef.current.onError?.(message);
      },
      onConnectionChange: (connected, attempt) => {
        setState((previous) => {
          if (previous.status === 'done' || previous.status === 'error') {
            return { ...previous, socketConnected: connected };
          }
          return {
            ...previous,
            socketConnected: connected,
            message: connected
              ? 'Detection running'
              : attempt > 0
                ? `Stream interrupted — reconnecting (attempt ${attempt})`
                : previous.message,
          };
        });
      },
    });

    return dispose;
  }, [jobId]);

  return state;
}

export function isTerminal(status: JobStatus): boolean {
  return status === 'done' || status === 'error';
}
