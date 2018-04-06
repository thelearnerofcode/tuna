#[derive(Debug, Clone)]
pub enum Tree {
    Branch { nodes: Vec<Tree> },
    Atom(String),
}

impl Tree {
    pub fn as_atom(&self) -> Option<String> {
        match *self {
            Tree::Atom(ref string) => Some(string.clone()),
            _ => None,
        }
    }

    pub fn get_branches(&self) -> Option<&[Tree]> {
        match *self {
            Tree::Branch { ref nodes } => Some(nodes),
            _ => None,
        }
    }

    pub fn to_string(&self) -> String {
        match *self {
            Tree::Atom(ref string) => string.clone(),
            Tree::Branch { ref nodes } => {
                let mut string = "(".to_owned();
                for node in nodes {
                    string.push_str(&node.to_string());
                    string.push(' ');
                }

                if nodes.len() != 0 {
                    string.pop();
                }

                string.push_str(")");
                string
            }
        }
    }

    pub fn to_string_pretty(&self) -> String {
        self.impl_to_string_pretty(0)
    }

    fn impl_to_string_pretty(&self, depth: usize) -> String {
        match *self {
            Tree::Atom(ref string) => string.clone(),
            Tree::Branch { ref nodes } => {
                let mut string = "(".to_owned();

                if let Some((first, nodes)) = nodes.split_first() {
                    string.push_str(&first.impl_to_string_pretty(depth + 1));

                    for node in nodes {
                        let node_string = node.impl_to_string_pretty(depth + 1);

                        if (node_string.len() > 20 && nodes.len() > 2) || node_string.len() > 40 {
                            string.push('\n');

                            for _ in 0..depth {
                                string.push_str("   ");
                            }
                        } else {
                            string.push(' ')
                        }
                        string.push_str(&node_string);
                    }
                }

                string.push_str(")");
                string
            }
        }
    }
}

impl Tree {
    fn from_tokens_impl(root_tree: bool, could_be_alone: bool, tokens: &[Token]) -> Tree {
        let mut nodes = Vec::new();
        let mut position = 0;

        while let Some(token) = tokens.iter().nth(position) {
            match *token {
                Token::RightParenthesis => {
                    let mut parenthesis_depth = 1;
                    let mut start_position = position + 1;
                    let mut end_position = position + 1;

                    'parenthesis_loop: loop {
                        match tokens.iter().nth(end_position) {
                            Some(token) => match *token {
                                Token::RightParenthesis => {
                                    parenthesis_depth += 1;
                                }
                                Token::LeftParenthesis => {
                                    parenthesis_depth -= 1;

                                    if parenthesis_depth == 0 {
                                        break 'parenthesis_loop;
                                    }
                                }
                                _ => {}
                            },
                            None => {
                                break 'parenthesis_loop;
                            }
                        }
                        end_position += 1;
                    }

                    nodes.push(Tree::from_tokens_impl(
                        false,
                        false,
                        &tokens[start_position..end_position],
                    ));
                    position = end_position;
                }
                Token::LeftParenthesis => {}
                Token::Atom(ref atom_string) => {
                    nodes.push(Tree::Atom(atom_string.clone()));
                }
            }
            position += 1;
        }

        if nodes.len() == 1 && (could_be_alone || root_tree) {
            nodes[0].clone()
        } else {
            Tree::Branch { nodes }
        }
    }

    pub fn from_tokens(tokens: &[Token]) -> Tree {
        Tree::from_tokens_impl(true, false, tokens)
    }
}

#[derive(Debug)]
pub enum Token {
    LeftParenthesis,
    RightParenthesis,
    Atom(String),
}

pub fn tokenize(source: &str) -> Vec<Token> {
    let mut position = 0;
    let mut tokens = Vec::new();

    while let Some(char) = source.chars().nth(position) {
        match char {
            ')' => tokens.push(Token::LeftParenthesis),
            '(' => tokens.push(Token::RightParenthesis),
            ' ' | '\n' => {}
            c => {
                let mut atom = String::new();
                let mut atom_position = position;

                'atom_loop: loop {
                    match source.chars().nth(atom_position) {
                        Some(char) => match char {
                            ')' | '(' | ' ' | '\n' => {
                                position = atom_position - 1;
                                break 'atom_loop;
                            }
                            c => {
                                atom_position += 1;
                                atom.push(c)
                            }
                        },
                        None => {
                            position = atom_position - 1;
                            break 'atom_loop;
                        }
                    }
                }

                tokens.push(Token::Atom(atom))
            }
        }

        position += 1;
    }
    tokens
}
