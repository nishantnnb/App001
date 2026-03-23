/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { 
  Menu, 
  Maximize2, 
  Minimize2, 
  RotateCcw, 
  RotateCw, 
  Play, 
  Pause, 
  FolderOpen, 
  X,
  Settings2,
  Activity,
  Upload,
  Bird,
  MapPin
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { get, set } from 'idb-keyval';
import { birdNetAnalyzer, BirdNetResult } from './lib/birdnet';

const FREQ_LIMIT = 15000;

export default function App() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [fftSize, setFftSize] = useState(1024);
  const [overlap, setOverlap] = useState(50);

  // BirdNET State
  const [modelLoaded, setModelLoaded] = useState(false);
  const [metaModelLoaded, setMetaModelLoaded] = useState(false);
  const [labelsLoaded, setLabelsLoaded] = useState(false);
  const [isInitializingModels, setIsInitializingModels] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);
  const [analysisResults, setAnalysisResults] = useState<BirdNetResult[]>([]);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [minConfidence, setMinConfidence] = useState(0.2);
  
  // Location State
  const [isLocationEnabled, setIsLocationEnabled] = useState(false);
  const [locationData, setLocationData] = useState<{lat: number, lon: number, week: number} | null>(null);
  const [metaProbabilities, setMetaProbabilities] = useState<Float32Array | null>(null);

  useEffect(() => {
    if (isLocationEnabled && locationData && metaModelLoaded) {
      birdNetAnalyzer.getMetaProbabilities(locationData).then(setMetaProbabilities);
    } else {
      setMetaProbabilities(null);
    }
  }, [isLocationEnabled, locationData, metaModelLoaded]);

  const displayResults = React.useMemo(() => {
    if (!analysisResults.length) return [];
    
    // 1. Apply penalty
    const penalized = analysisResults.map(res => {
      let conf = res.confidence;
      if (isLocationEnabled && metaProbabilities) {
        const idx = birdNetAnalyzer.getLabels().indexOf(res.label);
        if (idx >= 0) {
          conf = conf * metaProbabilities[idx];
        }
      }
      return { ...res, confidence: conf };
    });

    // 2. Group by start time and pick max
    const grouped = new Map<number, BirdNetResult>();
    for (const res of penalized) {
      if (res.confidence >= minConfidence) {
        const existing = grouped.get(res.start);
        if (!existing || res.confidence > existing.confidence) {
          grouped.set(res.start, res);
        }
      }
    }

    // 3. Sort by start time
    return Array.from(grouped.values()).sort((a, b) => a.start - b.start);
  }, [analysisResults, isLocationEnabled, metaProbabilities, minConfidence]);

  const audioRef = useRef<HTMLAudioElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sourceRef = useRef<MediaElementAudioSourceNode | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const fsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const animationIdRef = useRef<number | null>(null);
  const tempCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const tempFsCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const stopTimeRef = useRef<number | null>(null);
  const lastDrawnTimeRef = useRef<number>(-1);
  const abortControllerRef = useRef<AbortController | null>(null);

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  const initAudio = useCallback(() => {
    if (!audioCtxRef.current && audioRef.current) {
      const AudioContextClass = (window.AudioContext || (window as any).webkitAudioContext);
      audioCtxRef.current = new AudioContextClass();
      analyserRef.current = audioCtxRef.current.createAnalyser();
      analyserRef.current.fftSize = fftSize;
      analyserRef.current.smoothingTimeConstant = 0.5;
      sourceRef.current = audioCtxRef.current.createMediaElementSource(audioRef.current);
      sourceRef.current.connect(analyserRef.current);
      analyserRef.current.connect(audioCtxRef.current.destination);
    }
  }, [fftSize]);

  const getHeatmapColor = (value: number) => {
    const normalized = value / 255;
    if (normalized < 0.1) return `rgb(0, 0, 0)`;
    if (normalized < 0.3) return `rgb(0, 0, ${Math.floor(normalized * 3.3 * 100)})`;
    if (normalized < 0.5) return `rgb(${Math.floor((normalized - 0.3) * 5 * 180)}, 0, 100)`;
    if (normalized < 0.8) return `rgb(200, ${Math.floor((normalized - 0.5) * 3.3 * 200)}, 50)`;
    return `rgb(255, 255, ${Math.floor(150 + (normalized - 0.8) * 5 * 105)})`;
  };

  const renderSpectrogram = useCallback(() => {
    const draw = () => {
      animationIdRef.current = requestAnimationFrame(draw);
      if (!analyserRef.current || !audioCtxRef.current) return;

      const bufferLength = analyserRef.current.frequencyBinCount;
      const dataArray = new Uint8Array(bufferLength);
      analyserRef.current.getByteFrequencyData(dataArray);

      const scrollStep = 2 * (1 - (overlap / 100));
      const sampleRate = audioCtxRef.current.sampleRate;
      const maxBinIndex = Math.floor((FREQ_LIMIT * analyserRef.current.fftSize) / sampleRate);

      const drawToCanvas = (canvas: HTMLCanvasElement, tempCanvas: HTMLCanvasElement) => {
        const ctx = canvas.getContext('2d');
        const tCtx = tempCanvas.getContext('2d');
        if (!ctx || !tCtx) return;

        tCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
        tCtx.drawImage(canvas, 0, 0);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(tempCanvas, -scrollStep, 0);

        const barWidth = Math.ceil(scrollStep) || 1;
        const sliceX = canvas.width - barWidth;
        
        const timelineHeight = 20;
        const specHeight = canvas.height - timelineHeight;
        const barHeight = specHeight / maxBinIndex;

        for (let i = 0; i < maxBinIndex; i++) {
          ctx.fillStyle = getHeatmapColor(dataArray[i]);
          const y = specHeight - (i * barHeight);
          ctx.fillRect(sliceX, y, barWidth, Math.ceil(barHeight));
        }

        // Draw timeline background
        ctx.fillStyle = '#0e0e13';
        ctx.fillRect(sliceX, specHeight, barWidth, timelineHeight);

        // Draw time ticks
        const currentTime = audioRef.current?.currentTime || 0;
        const currentIntSec = Math.floor(currentTime);
        
        if (currentIntSec !== lastDrawnTimeRef.current) {
          lastDrawnTimeRef.current = currentIntSec;
          ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
          ctx.fillRect(sliceX, specHeight, 1, 5); // Tick mark
          
          ctx.font = '9px "Inter", sans-serif';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
          ctx.textAlign = 'right';
          ctx.fillText(formatTime(currentIntSec), sliceX - 2, canvas.height - 4);
        }
      };

      if (canvasRef.current && tempCanvasRef.current) {
        drawToCanvas(canvasRef.current, tempCanvasRef.current);
      }
      if (isFullscreen && fsCanvasRef.current && tempFsCanvasRef.current) {
        drawToCanvas(fsCanvasRef.current, tempFsCanvasRef.current);
      }
    };
    draw();
  }, [overlap, isFullscreen]);

  useEffect(() => {
    if (isPlaying) {
      renderSpectrogram();
    } else if (animationIdRef.current) {
      cancelAnimationFrame(animationIdRef.current);
    }
    return () => {
      if (animationIdRef.current) cancelAnimationFrame(animationIdRef.current);
    };
  }, [isPlaying, renderSpectrogram]);

  useEffect(() => {
    const resize = () => {
      if (canvasRef.current && tempCanvasRef.current) {
        canvasRef.current.width = canvasRef.current.clientWidth;
        canvasRef.current.height = canvasRef.current.clientHeight;
        tempCanvasRef.current.width = canvasRef.current.width;
        tempCanvasRef.current.height = canvasRef.current.height;
      }
      if (fsCanvasRef.current && tempFsCanvasRef.current) {
        fsCanvasRef.current.width = fsCanvasRef.current.clientWidth;
        fsCanvasRef.current.height = fsCanvasRef.current.clientHeight;
        tempFsCanvasRef.current.width = fsCanvasRef.current.width;
        tempFsCanvasRef.current.height = fsCanvasRef.current.height;
      }
    };
    window.addEventListener('resize', resize);
    resize();
    return () => window.removeEventListener('resize', resize);
  }, [isFullscreen]);

  useEffect(() => {
    if (!tempCanvasRef.current) tempCanvasRef.current = document.createElement('canvas');
    if (!tempFsCanvasRef.current) tempFsCanvasRef.current = document.createElement('canvas');
    
    // Load models and labels from IndexedDB
    const loadFromDB = async () => {
      try {
        const modelBuffer = await get('birdnet_model');
        const metaModelBuffer = await get('birdnet_meta_model');
        const labelsText = await get('birdnet_labels');
        
        if (modelBuffer) {
          const modelFile = new File([modelBuffer], 'audio-model.tflite', { type: 'application/octet-stream' });
          await birdNetAnalyzer.loadModel(modelFile);
          setModelLoaded(true);
        }
        
        if (metaModelBuffer) {
          const metaModelFile = new File([metaModelBuffer], 'meta-model.tflite', { type: 'application/octet-stream' });
          await birdNetAnalyzer.loadMetaModel(metaModelFile);
          setMetaModelLoaded(true);
        }
        
        if (labelsText) {
          const labelsFile = new File([labelsText], 'labels.txt', { type: 'text/plain' });
          await birdNetAnalyzer.loadLabels(labelsFile);
          setLabelsLoaded(true);
        }
      } catch (err) {
        console.error('Error loading from IndexedDB:', err);
      } finally {
        setIsInitializingModels(false);
      }
    };
    
    loadFromDB();
  }, []);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && audioRef.current) {
      setAudioFile(file);
      const url = URL.createObjectURL(file);
      audioRef.current.src = url;
      setProgress(0);
      stopTimeRef.current = null;
      lastDrawnTimeRef.current = -1;
      if (audioCtxRef.current?.state === 'suspended') {
        audioCtxRef.current.resume();
      }
    }
  };

  const togglePlay = () => {
    if (!audioRef.current?.src) {
      fileInputRef.current?.click();
      return;
    }
    initAudio();
    if (audioCtxRef.current?.state === 'suspended') audioCtxRef.current.resume();

    if (audioRef.current.paused) {
      // Stop analysis if playing
      if (isAnalyzing && abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }

      stopTimeRef.current = null; // Clear auto-stop on manual play
      audioRef.current.play();
      setIsPlaying(true);
    } else {
      audioRef.current.pause();
      setIsPlaying(false);
    }
  };

  const skip = (amount: number) => {
    if (audioRef.current) {
      audioRef.current.currentTime = Math.max(0, Math.min(audioRef.current.duration, audioRef.current.currentTime + amount));
      stopTimeRef.current = null;
      lastDrawnTimeRef.current = -1;
    }
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (audioRef.current?.duration) {
      const val = parseFloat(e.target.value);
      audioRef.current.currentTime = (val / 100) * audioRef.current.duration;
      setProgress(val);
      stopTimeRef.current = null;
      lastDrawnTimeRef.current = -1;
    }
  };

  const onTimeUpdate = () => {
    if (audioRef.current?.duration) {
      setProgress((audioRef.current.currentTime / audioRef.current.duration) * 100);
      if (stopTimeRef.current !== null && audioRef.current.currentTime >= stopTimeRef.current) {
        audioRef.current.pause();
        setIsPlaying(false);
        stopTimeRef.current = null;
      }
    }
  };

  const onEnded = () => {
    setIsPlaying(false);
  };

  const handleModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        await birdNetAnalyzer.loadModel(file);
        setModelLoaded(true);
        // Save to IndexedDB
        const buffer = await file.arrayBuffer();
        await set('birdnet_model', buffer);
      } catch (err) {
        console.error(err);
      }
    }
  };

  const handleMetaModelUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        await birdNetAnalyzer.loadMetaModel(file);
        setMetaModelLoaded(true);
        // Save to IndexedDB
        const buffer = await file.arrayBuffer();
        await set('birdnet_meta_model', buffer);
      } catch (err) {
        console.error(err);
      }
    }
  };

  const handleLabelsUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      try {
        await birdNetAnalyzer.loadLabels(file);
        setLabelsLoaded(true);
        // Save to IndexedDB
        const text = await file.text();
        await set('birdnet_labels', text);
      } catch (err) {
        console.error(err);
      }
    }
  };

  const handleFolderUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    const fileArray = Array.from(files);
    
    // Find files based on fixed names
    const modelFile = fileArray.find(f => f.name === 'audio-model.tflite');
    const metaModelFile = fileArray.find(f => f.name === 'meta-model.tflite');
    const labelsFile = fileArray.find(f => f.name.endsWith('.txt'));

    let loadedModel = false;
    let loadedMetaModel = false;
    let loadedLabels = false;

    if (modelFile) {
      try {
        await birdNetAnalyzer.loadModel(modelFile);
        setModelLoaded(true);
        const buffer = await modelFile.arrayBuffer();
        await set('birdnet_model', buffer);
        loadedModel = true;
      } catch (err) {
        console.error('Error loading audio model from folder:', err);
      }
    }

    if (metaModelFile) {
      try {
        await birdNetAnalyzer.loadMetaModel(metaModelFile);
        setMetaModelLoaded(true);
        const buffer = await metaModelFile.arrayBuffer();
        await set('birdnet_meta_model', buffer);
        loadedMetaModel = true;
      } catch (err) {
        console.error('Error loading meta model from folder:', err);
      }
    }

    if (labelsFile) {
      try {
        await birdNetAnalyzer.loadLabels(labelsFile);
        setLabelsLoaded(true);
        const text = await labelsFile.text();
        await set('birdnet_labels', text);
        loadedLabels = true;
      } catch (err) {
        console.error('Error loading labels from folder:', err);
      }
    }

    const found = [];
    if (loadedModel) found.push('Audio Model');
    if (loadedMetaModel) found.push('Meta Model');
    if (loadedLabels) found.push('Labels');
    
    if (found.length === 0) {
      alert("Could not find any valid models (audio-model.tflite, meta-model.tflite) or labels (.txt) in the selected folder.");
    } else {
      alert(`Successfully loaded: ${found.join(', ')}`);
    }
  };

  const toggleLocation = () => {
    if (isLocationEnabled) {
      setIsLocationEnabled(false);
      setLocationData(null);
    } else {
      if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser");
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (position) => {
          const now = new Date();
          // Calculate week of year (1-48 approx)
          const month = now.getMonth(); // 0-11
          const day = now.getDate(); // 1-31
          const weekOfMonth = Math.min(4, Math.ceil(day / 7.5)); // roughly 1-4
          const week = (month * 4) + weekOfMonth; // 1-48

          setLocationData({
            lat: position.coords.latitude,
            lon: position.coords.longitude,
            week: week
          });
          setIsLocationEnabled(true);
        },
        (error) => {
          console.error("Error getting location:", error);
          alert("Unable to retrieve your location. Please check permissions.");
        }
      );
    }
  };

  const runAnalysis = async () => {
    if (!audioFile) return;
    
    if (isAnalyzing) {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
        abortControllerRef.current = null;
      }
      return;
    }

    // Stop playback if analyzing
    if (isPlaying && audioRef.current) {
      audioRef.current.pause();
      setIsPlaying(false);
    }

    setIsAnalyzing(true);
    setAnalysisProgress(0);
    setAnalysisError(null);
    
    abortControllerRef.current = new AbortController();
    
    try {
      const results = await birdNetAnalyzer.analyzeAudio(
        audioFile, 
        (prog) => setAnalysisProgress(prog),
        abortControllerRef.current.signal
      );
      setAnalysisResults(results);
      if (results.length === 0) {
        setAnalysisError("No birds detected with confidence > 10%.");
      }
    } catch (err: any) {
      if (err.message === 'Analysis cancelled by user.') {
        setAnalysisError("Analysis stopped.");
      } else {
        console.error(err);
        setAnalysisError(err.message || "An unknown error occurred during analysis.");
      }
    } finally {
      setIsAnalyzing(false);
      abortControllerRef.current = null;
    }
  };

  return (
    <div className="h-screen flex flex-col relative overflow-hidden">
      <audio 
        ref={audioRef} 
        crossOrigin="anonymous" 
        onTimeUpdate={onTimeUpdate}
        onEnded={onEnded}
      />
      <input 
        type="file" 
        ref={fileInputRef} 
        className="hidden" 
        accept=".wav,.mp3,.flac,.aiff"
        onChange={handleFileChange}
        disabled={isAnalyzing}
      />

      {/* Side Menu */}
      <AnimatePresence>
        {isMenuOpen && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsMenuOpen(false)}
              className="fixed inset-0 bg-black/60 z-[55] backdrop-blur-sm"
            />
            <motion.div 
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed inset-y-0 left-0 w-80 z-[60] bg-[#0e0e13] border-r border-white/10 p-6 flex flex-col gap-8 shadow-2xl"
            >
              <div className="flex justify-between items-center">
                <h2 className="font-display text-[#a9ffdf] font-bold text-xl tracking-tight">Settings</h2>
                <button onClick={() => setIsMenuOpen(false)} className="text-white/60 hover:text-white transition-colors">
                  <X size={24} />
                </button>
              </div>
              <div className="space-y-6">
                <div className="space-y-4">
                  <div className="flex items-center gap-2 text-[#00d4ec]">
                    <Bird size={16} />
                    <span className="font-display text-[10px] uppercase tracking-widest font-bold">BirdNET Settings</span>
                  </div>
                  <div className="space-y-3">
                    {isInitializingModels ? (
                      <div className="p-3 rounded-xl border border-white/10 bg-white/5 text-center text-white/40 text-sm font-display animate-pulse">
                        Loading saved models...
                      </div>
                    ) : (
                      <>
                        <label className={`flex items-center justify-between p-3 rounded-xl border border-[#00FFC8]/30 bg-[#00FFC8]/10 transition-colors ${isAnalyzing ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer hover:bg-[#00FFC8]/20'}`}>
                          <div className="flex items-center gap-3">
                            <FolderOpen size={16} className="text-[#00FFC8]" />
                            <span className="font-display text-sm text-[#00FFC8] font-bold">Load Folder (Models + Labels)</span>
                          </div>
                          <input type="file" {...{webkitdirectory: "true", directory: "true"}} className="hidden" onChange={handleFolderUpload} disabled={isAnalyzing} />
                        </label>
                      </>
                    )}
                  </div>
                  <div className="space-y-2 mt-4">
                    <div className="flex justify-between">
                      <label className="block font-display text-[10px] text-white/40 tracking-widest uppercase font-bold">Min Confidence</label>
                      <span className="text-[10px] text-[#00FFC8] font-bold">{minConfidence.toFixed(2)}</span>
                    </div>
                    <input 
                      type="range" 
                      min="0.05" max="0.95" step="0.05"
                      value={minConfidence}
                      onChange={(e) => setMinConfidence(parseFloat(e.target.value))}
                      className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-[#00FFC8]"
                    />
                  </div>
                </div>

                <div className="flex items-center gap-2 text-[#a9ffdf] mt-6">
                  <Settings2 size={16} />
                  <span className="font-display text-[10px] uppercase tracking-widest font-bold">Spectrogram Settings</span>
                </div>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <label className="block font-display text-[10px] text-white/40 tracking-widest uppercase font-bold">FFT Size</label>
                    <select 
                      value={fftSize}
                      onChange={(e) => {
                        const val = parseInt(e.target.value);
                        setFftSize(val);
                        if (analyserRef.current) analyserRef.current.fftSize = val;
                      }}
                      className="w-full bg-white/5 border border-white/10 rounded-lg py-2 px-3 text-sm text-white focus:outline-none focus:border-[#00FFC8]/50 font-display"
                    >
                      {[512, 1024, 2048, 4096, 8192].map(size => (
                        <option key={size} value={size}>{size}</option>
                      ))}
                    </select>
                  </div>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <label className="block font-display text-[10px] text-white/40 tracking-widest uppercase font-bold">Overlap %</label>
                      <span className="text-[10px] text-[#00FFC8] font-bold">{overlap}%</span>
                    </div>
                    <input 
                      type="range" 
                      min="0" max="90" step="10"
                      value={overlap}
                      onChange={(e) => setOverlap(parseInt(e.target.value))}
                      className="w-full h-1 bg-white/10 rounded-lg appearance-none cursor-pointer accent-[#00FFC8]"
                    />
                  </div>
                </div>
              </div>
              <div className="mt-auto opacity-20 text-[10px] uppercase tracking-[0.2em] font-display">
                Engine Build v1.1.0
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Fullscreen View */}
      <AnimatePresence>
        {isFullscreen && (
          <motion.div 
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 z-[100] bg-[#0e0e13] flex flex-col"
          >
            <div className="relative flex-1">
              <canvas ref={fsCanvasRef} className="absolute inset-0 w-full h-full" />
              <div className="absolute left-6 top-0 bottom-0 flex flex-col justify-between py-12 font-display text-xs text-white/40 tracking-widest uppercase pointer-events-none z-30 font-bold">
                {['15', '10', '5', '2', '0.5', '0'].map((tick, i) => (
                  <span key={tick} className={i === 0 || i === 5 ? '' : `translate-y-[${i * 15}%]`}>{tick}</span>
                ))}
              </div>
              <button 
                onClick={() => setIsFullscreen(false)}
                className="absolute top-6 right-6 z-50 w-12 h-12 rounded-full bg-white/10 backdrop-blur-md border border-white/10 flex items-center justify-center text-[#a9ffdf] active:scale-90 transition-transform"
              >
                <Minimize2 size={24} />
              </button>
            </div>
            <div className="bg-black/80 backdrop-blur-xl border-t border-white/5 p-6 space-y-4">
              <input 
                type="range" 
                className="progress-slider"
                value={progress}
                onChange={handleSeek}
                disabled={isAnalyzing}
              />
              <div className="flex items-center justify-center gap-12">
                <button onClick={() => skip(-10)} disabled={isAnalyzing} className="text-white/60 active:scale-90 transition-transform disabled:opacity-50 disabled:cursor-not-allowed"><RotateCcw size={32} /></button>
                <button 
                  onClick={togglePlay}
                  disabled={isAnalyzing}
                  className="w-16 h-16 rounded-full bg-[#00FFC8] flex items-center justify-center text-[#0e0e13] active:scale-95 transition-transform disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isPlaying ? <Pause size={32} fill="currentColor" /> : <Play size={32} fill="currentColor" className="ml-1" />}
                </button>
                <button onClick={() => skip(10)} disabled={isAnalyzing} className="text-white/60 active:scale-90 transition-transform disabled:opacity-50 disabled:cursor-not-allowed"><RotateCw size={32} /></button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Top Bar */}
      <header className="fixed top-0 w-full z-50 bg-[#0e0e13]/70 backdrop-blur-xl flex justify-between items-center px-6 h-16 border-b border-white/5">
        <div className="flex items-center gap-4">
          <button onClick={() => setIsMenuOpen(true)} className="text-[#a9ffdf] transition-transform active:scale-95">
            <Menu size={24} />
          </button>
          <h1 className="font-display tracking-tighter font-bold text-xl text-[#a9ffdf]">Spectrolipi</h1>
        </div>
        <div className="flex items-center gap-4">
          <button onClick={() => setIsFullscreen(true)} className="text-[#a9ffdf] active:scale-95 transition-transform">
            <Maximize2 size={24} />
          </button>
          <div className="h-8 w-8 rounded-full bg-[#1f1f26] flex items-center justify-center border border-white/10 overflow-hidden">
            <img 
              src="https://lh3.googleusercontent.com/aida-public/AB6AXuA-aey1RwcK6YJE3J4MmjIdbOIB5Vjs1D_C24PunkF13P74Ewo1BCpx9rT9AOrP7GNCqcuJspdoc_T1aczvmaVU5IcxT4js8-LqM-tv9TsMb1PfYakxVyTVBkZ14kb54iLn_AntRwKY21LooHEbj0ijcY8s26VEoRu_Lrj8nYcvtmbb_sastTGUXNbUwaU3XfNOF9gXbReCXc4X0vCfKswkwucQl-wCAISbuI4179zU-JDyrr14Apw6joMWiByqcvVPmv3W4yHNyjw" 
              alt="Avatar" 
              className="w-full h-full object-cover"
              referrerPolicy="no-referrer"
            />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 mt-16 relative flex flex-col items-center p-0 overflow-y-auto overflow-x-hidden">
        <div className="absolute top-1/4 -left-20 w-96 h-96 bg-[#00FFC8]/5 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute bottom-1/4 -right-20 w-96 h-96 bg-[#00d4ec]/5 rounded-full blur-[120px] pointer-events-none" />

        {/* Spectrogram Container */}
        <div 
          className={`w-full relative glass-panel border-b border-white/10 pulse-glow ${isPlaying ? 'running' : ''}`}
          style={{ height: '40vh' }}
        >
          <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
          <div className="absolute left-4 top-0 bottom-0 flex flex-col justify-between py-6 font-display text-[10px] text-white/40 tracking-widest uppercase pointer-events-none z-30 font-bold">
            {['15', '10', '5', '2', '0.5', '0'].map((tick) => (
              <span key={tick}>{tick}</span>
            ))}
          </div>
        </div>

        {/* Progress Slider */}
        <div className="w-full bg-white/5">
          <input 
            type="range" 
            className="progress-slider"
            value={progress}
            onChange={handleSeek}
            disabled={isAnalyzing}
          />
        </div>

        {/* File Name Display */}
        {audioFile && (
          <div className="w-full flex justify-center py-1.5 bg-white/5 border-b border-white/10">
            <span className="font-mono text-[10px] text-white/40 tracking-wider truncate max-w-md px-4">
              {audioFile.name}
            </span>
          </div>
        )}

        {/* Controls */}
        <div className="w-full flex flex-col items-center gap-6 py-6 px-4 sm:px-6">
          <div className="flex items-center gap-6 sm:gap-8">
            <button 
              onClick={() => fileInputRef.current?.click()}
              disabled={isAnalyzing}
              className="text-white/60 hover:text-[#a9ffdf] transition-colors active:scale-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center w-8 h-8"
              title="Load Audio File"
            >
              <FolderOpen size={24} />
            </button>
            <button 
              onClick={runAnalysis}
              disabled={!modelLoaded || !labelsLoaded || !audioFile}
              className={`transition-colors active:scale-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center w-8 h-8 ${isAnalyzing ? 'text-red-400' : 'text-[#00d4ec] hover:text-white'}`}
              title={isAnalyzing ? `Stop Analysis` : "Analyze Audio"}
            >
              {isAnalyzing ? (
                <span className="text-sm font-mono font-bold">{analysisProgress}%</span>
              ) : (
                <Activity size={24} />
              )}
            </button>
            <button onClick={() => skip(-10)} disabled={isAnalyzing} className="text-white/60 hover:text-[#a9ffdf] transition-colors active:scale-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center w-8 h-8">
              <RotateCcw size={24} />
            </button>
            <button 
              onClick={togglePlay}
              disabled={isAnalyzing}
              className="text-[#00FFC8] hover:text-white transition-colors active:scale-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center w-8 h-8"
            >
              {isPlaying ? <Pause size={24} /> : <Play size={24} />}
            </button>
            <button onClick={() => skip(10)} disabled={isAnalyzing} className="text-white/60 hover:text-[#a9ffdf] transition-colors active:scale-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center w-8 h-8">
              <RotateCw size={24} />
            </button>
            <button 
              onClick={toggleLocation}
              disabled={isAnalyzing || !metaModelLoaded}
              className={`transition-colors active:scale-90 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center w-8 h-8 ${isLocationEnabled ? 'text-[#00FFC8]' : 'text-white/60 hover:text-[#a9ffdf]'}`}
              title={!metaModelLoaded ? "Load meta-model.tflite to enable location filtering" : "Toggle Location Filtering"}
            >
              <MapPin size={24} />
            </button>
          </div>

          {analysisError && (
            <div className="w-full max-w-md p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-red-400 text-xs font-mono text-center">
              {analysisError}
            </div>
          )}

          {displayResults.length > 0 && (
            <div className="w-full max-w-md space-y-4 mt-4">
              <h3 className="font-display text-[10px] text-[#00FFC8] tracking-widest uppercase font-bold">Detections ({displayResults.length})</h3>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-2 custom-scrollbar">
                {displayResults.map((res, idx) => (
                  <div key={idx} className="py-1.5 px-3 rounded-md bg-white/5 border border-white/5 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2 truncate">
                      <span className="font-bold text-xs text-white truncate">{res.label.split('_').pop()}</span>
                      <span className="text-[10px] text-[#00FFC8] font-mono">{(res.confidence * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex items-center gap-3 shrink-0">
                      <span className="text-[10px] text-white/40 font-mono">{formatTime(res.start)}-{formatTime(res.end)}</span>
                      <button 
                        onClick={() => {
                          if (audioRef.current) {
                            initAudio();
                            audioRef.current.currentTime = res.start;
                            stopTimeRef.current = res.end; // Auto-stop at the end of the detection
                            lastDrawnTimeRef.current = -1;
                            if (audioCtxRef.current?.state === 'suspended') audioCtxRef.current.resume();
                            if (!isPlaying) {
                              audioRef.current.play();
                              setIsPlaying(true);
                            }
                          }
                        }}
                        disabled={isAnalyzing}
                        className="hover:text-white transition-colors flex items-center gap-1 text-[10px] bg-white/10 px-2 py-1 rounded active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        <Play size={10} />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
