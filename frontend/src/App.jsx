import { useCallback, useEffect, useRef, useState } from 'react'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// STATUS: idle -> loaded -> developing -> done | error
function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [status, setStatus] = useState('idle')
  const [caption, setCaption] = useState('')
  const [typedCaption, setTypedCaption] = useState('')
  const [errorMsg, setErrorMsg] = useState('')
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)
  const typeTimer = useRef(null)

  const reset = useCallback(() => {
    setFile(null)
    setPreviewUrl(null)
    setStatus('idle')
    setCaption('')
    setTypedCaption('')
    setErrorMsg('')
  }, [])

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f)
    setPreviewUrl(URL.createObjectURL(f))
    setStatus('loaded')
    setCaption('')
    setTypedCaption('')
    setErrorMsg('')
  }, [])

  const onDrop = useCallback(
    (e) => {
      e.preventDefault()
      setIsDragging(false)
      const f = e.dataTransfer.files?.[0]
      handleFile(f)
    },
    [handleFile],
  )

  const develop = useCallback(async () => {
    if (!file) return
    setStatus('developing')
    setErrorMsg('')
    try {
      const formData = new FormData()
      formData.append('file', file)
      const res = await fetch(`${API_URL}/caption`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const body = await res.json().catch(() => ({}))
        throw new Error(body.detail || `Request failed (${res.status})`)
      }
      const data = await res.json()
      setCaption(data.caption || '')
      setStatus('done')
    } catch (err) {
      setErrorMsg(err.message || 'Something went wrong developing this print.')
      setStatus('error')
    }
  }, [file])

  // Typewriter reveal once caption arrives
  useEffect(() => {
    if (status !== 'done' || !caption) return
    setTypedCaption('')
    let i = 0
    typeTimer.current = setInterval(() => {
      i += 1
      setTypedCaption(caption.slice(0, i))
      if (i >= caption.length) {
        clearInterval(typeTimer.current)
      }
    }, 28)
    return () => clearInterval(typeTimer.current)
  }, [status, caption])

  return (
    <div className="stage">
      <header className="header">
        <span className="header__eyebrow">VGG16 · LSTM</span>
        <h1 className="header__title">Print / Caption</h1>
        <p className="header__sub">Drop a photo in the tray. Watch the words develop.</p>
      </header>

      <main className="darkroom">
        <div
          className={`tray ${isDragging ? 'tray--dragging' : ''} ${previewUrl ? 'tray--loaded' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setIsDragging(true)
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={onDrop}
          onClick={() => !previewUrl && inputRef.current?.click()}
        >
          <div className="tray__liquid" aria-hidden="true" />
          {previewUrl ? (
            <img src={previewUrl} alt="Selected" className="tray__image" />
          ) : (
            <div className="tray__empty">
              <div className="tray__empty-icon">＋</div>
              <p>Click or drag a photo into the tray</p>
              <span className="tray__empty-hint">JPG or PNG</span>
            </div>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            hidden
            onChange={(e) => handleFile(e.target.files?.[0])}
          />
        </div>

        <div className="controls">
          {previewUrl && status !== 'developing' && (
            <button className="btn btn--primary" onClick={develop}>
              {status === 'done' ? 'Develop again' : 'Develop caption'}
            </button>
          )}
          {status === 'developing' && (
            <button className="btn btn--busy" disabled>
              <span className="pulse-dot" /> Developing…
            </button>
          )}
          {previewUrl && (
            <button className="btn btn--ghost" onClick={reset}>
              Clear tray
            </button>
          )}
        </div>

        {errorMsg && <p className="error-note">{errorMsg}</p>}

        {(status === 'developing' || status === 'done') && (
          <div className={`label-card ${status === 'done' ? 'label-card--lit' : ''}`}>
            <div className="label-card__pin" />
            {status === 'developing' ? (
              <p className="label-card__text label-card__text--waiting">
                exposing negative<span className="ellipsis" />
              </p>
            ) : (
              <p className="label-card__text">
                {typedCaption}
                <span className="cursor">|</span>
              </p>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <span>Local inference · VGG16 feature extractor + LSTM decoder</span>
      </footer>
    </div>
  )
}

export default App
