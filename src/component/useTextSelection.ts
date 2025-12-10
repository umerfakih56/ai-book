import { useEffect, useState } from "react";

export default function useTextSelection() {
  const [selectedText, setSelectedText] = useState<string | null>(null);

  useEffect(() => {
    function handleSelectionChange() {
      const sel = window.getSelection();
      if (!sel) return;
      const text = sel.toString().trim();
      if (text.length > 20) {
        setSelectedText(text);
      } else {
        setSelectedText(null);
      }
    }

    document.addEventListener("mouseup", handleSelectionChange);
    document.addEventListener("keyup", handleSelectionChange);

    return () => {
      document.removeEventListener("mouseup", handleSelectionChange);
      document.removeEventListener("keyup", handleSelectionChange);
    };
  }, []);

  return {
    selectedText,
    hasSelection: !!selectedText,
    clearSelection: () => setSelectedText(null),
    highlightSelection: () => {
      // Optional: implement visual highlighting via ranges
    },
  };
}