import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { API_BASE } from '@/lib/api';

export interface EmbeddingPoint {
  filename: string;
  coordinates: number[];
  embedding?: number[];
  embedding_dim?: number;
}

interface EmbeddingData {
  model: string;
  dataset: string;
  reduction_method: string;
  n_components: number;
  embeddings: Array<{
    filename: string;
    embedding: number[];
    embedding_dim: number;
  }>;
  reduced_embeddings?: EmbeddingPoint[];
  total_files: number;
  original_dimension: number;
}

interface EmbeddingContextType {
  embeddingData: EmbeddingData | null;
  isLoading: boolean;
  error: string | null;
  fetchEmbeddings: (
    model: string,
    dataset: string,
    files: EmbeddingFileReference[],
    reductionMethod?: string,
    nComponents?: number
  ) => Promise<void>;
  clearEmbeddings: () => void;
}

export type EmbeddingFileReference = string | {
  filename: string;
  file_path: string;
};

const EmbeddingContext = createContext<EmbeddingContextType | undefined>(undefined);

export const useEmbedding = () => {
  const context = useContext(EmbeddingContext);
  if (context === undefined) {
    throw new Error('useEmbedding must be used within an EmbeddingProvider');
  }
  return context;
};

export const EmbeddingProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [embeddingData, setEmbeddingData] = useState<EmbeddingData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const activeRequest = useRef<AbortController | null>(null);

  const fetchEmbeddings = useCallback(async (
    model: string,
    dataset: string,
    files: EmbeddingFileReference[],
    reductionMethod: string = 'pca',
    nComponents: number = 3
  ) => {
    if (!files || files.length === 0) {
      setError('No files provided');
      return;
    }

    activeRequest.current?.abort();
    const controller = new AbortController();
    activeRequest.current = controller;

    setEmbeddingData(null);
    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE}/inferences/embeddings`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        signal: controller.signal,
        body: JSON.stringify({
          model,
          dataset,
          files,
          reduction_method: reductionMethod,
          n_components: nComponents,
        }),
      });

      if (!response.ok) {
        throw new Error(`Failed to fetch embeddings: ${response.status}`);
      }

      const data = await response.json();
      if (activeRequest.current === controller) {
        setEmbeddingData(data);
      }
    } catch (err) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch embeddings';
      if (activeRequest.current === controller) setError(errorMessage);
      console.error('Error fetching embeddings:', err);
    } finally {
      if (activeRequest.current === controller) {
        activeRequest.current = null;
        setIsLoading(false);
      }
    }
  }, []);

  const clearEmbeddings = useCallback(() => {
    activeRequest.current?.abort();
    activeRequest.current = null;
    setEmbeddingData(null);
    setIsLoading(false);
    setError(null);
  }, []);

  return (
    <EmbeddingContext.Provider value={{
      embeddingData,
      isLoading,
      error,
      fetchEmbeddings,
      clearEmbeddings,
    }}>
      {children}
    </EmbeddingContext.Provider>
  );
};
