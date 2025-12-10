import React from "react";
import type { ReactNode } from "react";
import RAGChatWidget from "../component/RagChatWidget";



interface RootProps {
  children: ReactNode;
}

function Root({ children }: RootProps) {
  return (
    <>
      {children}
      <RAGChatWidget/>
    </>
  );
}

export default Root;
