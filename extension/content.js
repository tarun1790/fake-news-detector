// Veritas AI Content Extraction Script
(() => {
    // 1. Get the article title
    let title = document.title;
    
    // Attempt to grab clean headers
    const h1 = document.querySelector('h1');
    if (h1 && h1.innerText.trim().length > 5) {
        title = h1.innerText.trim();
    }

    // 2. Extract article body text
    // We want to extract main paragraph tags while ignoring navigation, comments, headers, footers, etc.
    const ignoredTags = ['header', 'footer', 'nav', 'aside', 'form', 'script', 'style', 'noscript', 'iframe'];
    
    // Helper to check if node or any of its ancestors are in ignoredTags
    function isIgnored(node) {
        let current = node;
        while (current && current !== document.body) {
            if (ignoredTags.includes(current.tagName.toLowerCase())) {
                return true;
            }
            // Check for common sidebar/ads class names
            const className = (current.className || '').toString().toLowerCase();
            const id = (current.id || '').toString().toLowerCase();
            if (className.includes('sidebar') || className.includes('menu') || className.includes('footer') || 
                className.includes('header') || className.includes('advertisement') || className.includes('widget') ||
                id.includes('sidebar') || id.includes('menu') || id.includes('footer') || id.includes('header') || 
                id.includes('ads')) {
                return true;
            }
            current = current.parentNode;
        }
        return false;
    }

    const paragraphs = document.querySelectorAll('p');
    let articleText = [];
    
    paragraphs.forEach(p => {
        if (!isIgnored(p)) {
            const text = p.innerText.trim();
            // Standard sentence heuristic: only include paragraphs that are of significant length
            if (text.length > 60 && text.split(/\s+/).length > 8) {
                articleText.push(text);
            }
        }
    });

    // Fallback: If no paragraphs found (e.g. some news platforms use divs/spans)
    if (articleText.length === 0) {
        const bodyContent = document.body.innerText;
        // Grab lines that look like paragraphs
        const lines = bodyContent.split('\n');
        lines.forEach(line => {
            const trimmed = line.trim();
            if (trimmed.length > 80 && trimmed.split(/\s+/).length > 12) {
                articleText.push(trimmed);
            }
        });
    }

    // Combine extracted paragraphs, limit to ~12,000 chars for efficiency
    let fullText = articleText.join('\n\n');
    if (fullText.length > 12000) {
        fullText = fullText.substring(0, 12000) + '... [Truncated for analysis]';
    }

    return {
        title: title,
        text: fullText,
        url: window.location.href
    };
})();
