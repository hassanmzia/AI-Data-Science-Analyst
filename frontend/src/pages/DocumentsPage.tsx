import React, { useState, useEffect, useCallback } from 'react';
import { FileSearch, Upload, Send, Loader2, FileText, Trash2 } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import ReactMarkdown from 'react-markdown';
import { documentApi } from '../services/api';
import { Document } from '../types';
import toast from 'react-hot-toast';

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<Document | null>(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<any>(null);
  const [querying, setQuerying] = useState(false);
  const [showUpload, setShowUpload] = useState(false);

  useEffect(() => { loadDocuments(); }, []);

  async function loadDocuments() {
    try {
      const res = await documentApi.list();
      setDocuments(res.data.results || res.data || []);
    } catch (e) { console.error(e); }
  }

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    for (const file of acceptedFiles) {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('name', file.name);
      try {
        await documentApi.upload(formData);
        toast.success(`Uploaded & indexed: ${file.name}`);
        loadDocuments();
      } catch (e: any) {
        toast.error(`Upload failed: ${e.response?.data?.error || e.message}`);
      }
    }
    setShowUpload(false);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'text/markdown': ['.md'],
    },
  });

  async function askQuestion() {
    if (!selectedDoc || !question) return;
    setQuerying(true);
    setAnswer(null);
    try {
      const res = await documentApi.query(selectedDoc.id, question);
      setAnswer(res.data);
    } catch (e: any) {
      toast.error(`Query failed: ${e.response?.data?.error || e.message}`);
    } finally { setQuerying(false); }
  }

  async function deleteDocument(id: string) {
    if (!window.confirm('Delete this document?')) return;
    try {
      await documentApi.delete(id);
      toast.success('Deleted');
      loadDocuments();
      if (selectedDoc?.id === id) { setSelectedDoc(null); setAnswer(null); }
    } catch { toast.error('Delete failed'); }
  }

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Documents (RAG)</h1>
          <p className="text-sm text-gray-500 mt-1">Upload documents and ask questions using retrieval-augmented generation</p>
        </div>
        <button onClick={() => setShowUpload(!showUpload)}
          className="px-4 py-2 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 flex items-center">
          <Upload size={16} className="mr-2" /> Upload Document
        </button>
      </div>

      {showUpload && (
        <div className="mb-6 bg-white rounded-xl p-6 border-2 border-dashed border-gray-300">
          <div {...getRootProps()} className="text-center cursor-pointer py-8">
            <input {...getInputProps()} />
            <FileText size={40} className="mx-auto text-gray-400 mb-3" />
            {isDragActive ? (
              <p className="text-primary-600">Drop document here...</p>
            ) : (
              <div>
                <p className="text-gray-600">Drag & drop documents, or click to select</p>
                <p className="text-xs text-gray-400 mt-1">Supports PDF, DOCX, TXT, Markdown</p>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Document list */}
        <div className="space-y-3">
          <h2 className="text-lg font-semibold text-gray-900">Documents</h2>
          {documents.map(doc => (
            <div key={doc.id} onClick={() => { setSelectedDoc(doc); setAnswer(null); }}
              className={`bg-white rounded-lg p-4 border cursor-pointer hover:shadow-sm ${
                selectedDoc?.id === doc.id ? 'border-primary-500' : 'border-gray-200'
              }`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center">
                  <FileText size={18} className="text-primary-500 mr-2" />
                  <div>
                    <h3 className="font-medium text-gray-900 text-sm">{doc.name}</h3>
                    <p className="text-xs text-gray-500">
                      {doc.doc_type} | {doc.is_indexed ? `${doc.chunk_count} chunks` : 'Not indexed'}
                    </p>
                  </div>
                </div>
                <button onClick={(e) => { e.stopPropagation(); deleteDocument(doc.id); }}
                  className="p-1 hover:bg-red-50 rounded">
                  <Trash2 size={14} className="text-gray-400 hover:text-red-500" />
                </button>
              </div>
            </div>
          ))}
          {documents.length === 0 && (
            <p className="text-center text-gray-500 py-6">No documents uploaded</p>
          )}
        </div>

        {/* Q&A area */}
        <div className="lg:col-span-2">
          {selectedDoc ? (
            <div className="bg-white rounded-xl border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Ask about: {selectedDoc.name}</h3>
              <div className="flex space-x-3 mb-6">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && askQuestion()}
                  placeholder="e.g., What are the different states of a process?"
                  className="flex-1 px-4 py-3 border rounded-lg text-sm"
                />
                <button onClick={askQuestion} disabled={querying || !question}
                  className="px-6 py-3 bg-primary-600 text-white rounded-lg text-sm hover:bg-primary-700 disabled:opacity-50">
                  {querying ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
                </button>
              </div>

              {answer && (
                <div className="space-y-4">
                  <div className="bg-gray-50 rounded-lg p-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Answer</h4>
                    <div className="prose prose-sm max-w-none">
                      <ReactMarkdown>{answer.answer}</ReactMarkdown>
                    </div>
                  </div>
                  {answer.sources && answer.sources.length > 0 && (
                    <details>
                      <summary className="cursor-pointer text-sm text-primary-600 font-medium">
                        View Sources ({answer.sources.length})
                      </summary>
                      <div className="mt-2 space-y-2">
                        {answer.sources.map((source: string, i: number) => (
                          <div key={i} className="bg-gray-50 p-3 rounded-lg text-xs text-gray-600">
                            {source.substring(0, 300)}...
                          </div>
                        ))}
                      </div>
                    </details>
                  )}
                </div>
              )}
            </div>
          ) : (
            <div className="bg-white rounded-xl border border-gray-200 p-12 text-center">
              <FileSearch size={40} className="mx-auto text-gray-300 mb-3" />
              <p className="text-gray-500">Select a document to ask questions</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
