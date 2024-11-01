type Author = {
  name: string;
  url?: string;
  notes?: string[];
}

type Institution = {
  name: string;
  url?: string;
  notes?: string[];
}

type Link = {
  url: string;
  name: string;
  icon?: string;
}

type Note = {
  symbol: string;
  text: string;
}

export { Author, Institution, Link, Note };