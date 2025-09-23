export const getLatestResult = async () => {
  const response = await fetch('/results/latest');
  if (!response.ok) {
    throw new Error('Failed to fetch latest result');
  }
  return response.json();
};

export const getResultSummary = async (itemName: string) => {
  const response = await fetch(`/results/${encodeURIComponent(itemName)}/summary`);
  if (!response.ok) {
    throw new Error('Failed to fetch result summary');
  }
  return response.json();
};