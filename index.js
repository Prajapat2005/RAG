import * as dotenv from 'dotenv';
dotenv.config();

import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';

const indexDocument = async () => {

    const PDF_PATH = "./cpbookspringer.pdf";
    const pdfLoader = new PDFLoader(PDF_PATH);
    const rawDocs = await pdfLoader.load();
    console.log("PDF loaded");

    const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 2000,
        chunkOverlap: 200,
    });
    const chunkedDocs = await textSplitter.splitDocuments(rawDocs);
    console.log("Chunking completed");

    const embeddings = new GoogleGenerativeAIEmbeddings({
        apiKey: process.env.GEMINI_API_KEY,
        model: 'text-embedding-004',
    });
    console.log("Embedding model configured");

    // Database configuration
    const pinecone = new Pinecone(); // automatically take api key from env variable named PINECONE_API_KEY
    const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

    // (chunking, embedding, DB) ---> langchain automatically embedding of chunks store to DB 
    await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
        pineconeIndex,
        maxConcurrency: 5,  // max for free
    });
    console.log("Data stored");

}

indexDocument();