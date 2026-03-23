import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-cpu';
import '@tensorflow/tfjs-backend-webgl';

// Expose tf to window for the tflite CDN script
(window as any).tf = tf;

export interface BirdNetResult {
  start: number;
  end: number;
  label: string;
  confidence: number;
}

export interface LocationData {
  lat: number;
  lon: number;
  week: number;
}

export class BirdNetAnalyzer {
  private model: any = null;
  private metaModel: any = null;
  private labels: string[] = [];
  private isInitializing = false;

  async loadModel(modelFile: File) {
    if (this.isInitializing) return;
    this.isInitializing = true;
    try {
      // Ensure tflite is loaded
      let tflite = (window as any).tflite;
      if (!tflite) {
        await new Promise<void>((resolve, reject) => {
          const script = document.createElement('script');
          script.src = 'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js';
          script.onload = () => resolve();
          script.onerror = () => reject(new Error('Failed to load TFLite script from CDN.'));
          document.head.appendChild(script);
        });
        tflite = (window as any).tflite;
      }

      if (!tflite) {
        throw new Error('TFLite library failed to initialize.');
      }
      
      tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/');
      const arrayBuffer = await modelFile.arrayBuffer();
      this.model = await tflite.loadTFLiteModel(arrayBuffer, { numThreads: 1 });
      console.log('BirdNET model loaded successfully. Inputs:', this.model.inputs);
    } catch (error) {
      console.error('Error loading BirdNET model:', error);
      throw error;
    } finally {
      this.isInitializing = false;
    }
  }

  async loadMetaModel(modelFile: File) {
    try {
      let tflite = (window as any).tflite;
      if (!tflite) {
        throw new Error('TFLite library not initialized.');
      }
      const arrayBuffer = await modelFile.arrayBuffer();
      this.metaModel = await tflite.loadTFLiteModel(arrayBuffer, { numThreads: 1 });
      console.log('BirdNET meta model loaded successfully. Inputs:', this.metaModel.inputs);
    } catch (error) {
      console.error('Error loading BirdNET meta model:', error);
      throw error;
    }
  }

  async loadLabels(labelsFile: File) {
    try {
      const text = await labelsFile.text();
      this.labels = text.split('\n').map(line => line.trim()).filter(line => line.length > 0);
      console.log(`Loaded ${this.labels.length} labels.`);
    } catch (error) {
      console.error('Error loading labels:', error);
      throw error;
    }
  }

  isReady() {
    return this.model !== null && this.labels.length > 0;
  }

  getLabels(): string[] {
    return this.labels;
  }

  async getMetaProbabilities(locationData: LocationData): Promise<Float32Array | null> {
    if (!this.metaModel) return null;
    try {
      const metaInput = tf.tensor2d([[locationData.lat, locationData.lon, locationData.week]], [1, 3], 'float32');
      const metaOutput = this.metaModel.predict({ [this.metaModel.inputs[0].name]: metaInput });
      
      let metaOutputTensor: tf.Tensor;
      if (metaOutput instanceof tf.Tensor) {
        metaOutputTensor = metaOutput;
      } else if (Array.isArray(metaOutput)) {
        metaOutputTensor = metaOutput[0];
      } else {
        metaOutputTensor = Object.values(metaOutput)[0] as tf.Tensor;
      }
      
      const rawMetaData = await metaOutputTensor.data();
      const metaProbabilities = new Float32Array(rawMetaData.length);
      
      let isLogits = false;
      for (let j = 0; j < rawMetaData.length; j++) {
        if (rawMetaData[j] < 0 || rawMetaData[j] > 1) {
          isLogits = true;
          break;
        }
      }
      
      for (let j = 0; j < rawMetaData.length; j++) {
        metaProbabilities[j] = isLogits ? 1 / (1 + Math.exp(-rawMetaData[j])) : rawMetaData[j];
      }
      
      metaOutputTensor.dispose();
      metaInput.dispose();
      console.log('Meta model probabilities computed successfully.');
      return metaProbabilities;
    } catch (e) {
      console.error('Error running meta model:', e);
      return null;
    }
  }

  async analyzeAudio(audioFile: File, onProgress?: (progress: number) => void, abortSignal?: AbortSignal): Promise<BirdNetResult[]> {
    if (!this.isReady() || !this.model) {
      throw new Error('Model or labels not loaded.');
    }

    let audioCtx: AudioContext;
    try {
      audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 48000 });
    } catch (e) {
      throw new Error('Failed to initialize AudioContext. Your browser might not support it.');
    }

    let channelData: Float32Array;
    try {
      const arrayBuffer = await audioFile.arrayBuffer();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
      channelData = audioBuffer.getChannelData(0); // Mono
    } catch (e) {
      throw new Error('Failed to decode audio file. Ensure it is a valid audio format.');
    }

    const sampleRate = 48000;
    let chunkSize = sampleRate * 3.0; // Default 3 seconds (144000 samples)
    let inputShape = [1, chunkSize];

    // Dynamically determine input shape from the model
    let inputType = 'float32';
    if (this.model.inputs && this.model.inputs.length > 0) {
      const shape = this.model.inputs[0].shape;
      inputType = this.model.inputs[0].dtype || 'float32';
      console.log('Model input shape:', shape, 'type:', inputType);
      if (shape && shape.length > 0) {
        inputShape = shape.map((s: number, idx: number) => {
          if (s > 0) return s;
          if (idx === 0) return 1; // Default batch size to 1
          return chunkSize; // Default dynamic sequence length to 144000
        });
        chunkSize = inputShape.reduce((a: number, b: number) => a * b, 1);
      }
    }

    const chunkDuration = chunkSize / sampleRate;
    const stepSize = chunkSize; // No overlap for simplicity
    const results: BirdNetResult[] = [];
    const totalChunks = Math.ceil(channelData.length / stepSize);

    if (totalChunks === 0) {
      throw new Error('Audio file is too short or empty.');
    }

    for (let i = 0; i < channelData.length; i += stepSize) {
      if (abortSignal?.aborted) {
        throw new Error('Analysis cancelled by user.');
      }

      const chunk = new Float32Array(chunkSize);
      const end = Math.min(i + chunkSize, channelData.length);
      
      // Scale from [-1.0, 1.0] to [-32768.0, 32767.0] as expected by BirdNET
      for (let j = i; j < end; j++) {
        chunk[j - i] = channelData[j] * 32767.0;
      }

      // Pad with zeros if chunk is smaller than expected
      if (end - i < chunkSize) {
        chunk.fill(0, end - i);
      }

      let outputTensor: tf.Tensor | null = null;

      try {
        outputTensor = tf.tidy(() => {
          let tensorData: Float32Array | Int32Array = chunk;
          if (inputType === 'int32') {
             tensorData = new Int32Array(chunk);
          }
          const inputTensor = tf.tensor(tensorData, inputShape, inputType as any);
          const inputName = this.model.inputs[0].name;
          const output = this.model.predict({ [inputName]: inputTensor });

          if (output instanceof tf.Tensor) {
            return output;
          } else if (Array.isArray(output)) {
            return output[0];
          } else {
            return Object.values(output)[0] as tf.Tensor;
          }
        });

        const outputData = await outputTensor.data();

        const predictions = [];
        const limit = Math.min(outputData.length, this.labels.length);
        for (let j = 0; j < limit; j++) {
          const val = outputData[j];
          const prob = 1 / (1 + Math.exp(-val));
          if (prob > 0.01) {
            predictions.push({ index: j, prob });
          }
        }

        predictions.sort((a, b) => b.prob - a.prob);
        const topPredictions = predictions.slice(0, 5);

        const startTime = i / sampleRate;
        for (const p of topPredictions) {
          results.push({
            start: startTime,
            end: startTime + chunkDuration,
            label: this.labels[p.index],
            confidence: p.prob,
          });
        }
      } catch (e: any) {
        console.error('Inference error at chunk', i, e);
        throw new Error(`Inference failed: ${e.message}. Model expected input shape: ${JSON.stringify(inputShape)}. Ensure your model accepts raw audio and not spectrograms.`);
      } finally {
        if (outputTensor) {
          outputTensor.dispose();
        }
      }

      if (onProgress) {
        onProgress(Math.min(100, Math.round(((i + stepSize) / channelData.length) * 100)));
      }

      // Yield to main thread to allow UI to update
      await new Promise(resolve => setTimeout(resolve, 10));
    }

    return results.sort((a, b) => a.start - b.start);
  }
}

export const birdNetAnalyzer = new BirdNetAnalyzer();
