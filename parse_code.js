const fs = require('fs');
const { parse } = require('@babel/parser');
const traverse = require('@babel/traverse').default;

function extractFunctions(ast) {
    const functions = new Map();
    // Structure: functions.set(uniqueName, { paramsCount: number });
    // uniqueName will be something like "functionName", "variableName", or "ClassName.methodName"

    function recordFunction(name, paramsCount) {
        // If name isn't found, generate a unique fallback
        const key = name || `anonymous_function_${functions.size}`;
        functions.set(key, { paramsCount });
    }

    traverse(ast, {
        FunctionDeclaration(path) {
            const name = path.node.id && path.node.id.name;
            const paramsCount = path.node.params.length;
            recordFunction(name, paramsCount);
        },
        ArrowFunctionExpression(path) {
            // For arrow functions, try to find if it's assigned to a variable or a class property
            if (path.parent.type === 'VariableDeclarator' && path.parent.id.type === 'Identifier') {
                const name = path.parent.id.name;
                const paramsCount = path.node.params.length;
                recordFunction(name, paramsCount);
            } else if (path.parent.type === 'ClassProperty' && path.parent.key.type === 'Identifier') {
                const classMethodName = path.parent.key.name;
                // Try to find class name
                let className = "UnknownClass";
                let current = path.parentPath;
                while (current && current.node && current.node.type !== 'ClassDeclaration') {
                    current = current.parentPath;
                }
                if (current && current.node && current.node.id && current.node.id.name) {
                    className = current.node.id.name;
                }
                const name = `${className}.${classMethodName}`;
                const paramsCount = path.node.params.length;
                recordFunction(name, paramsCount);
            } else {
                // Just treat as anonymous arrow function
                const paramsCount = path.node.params.length;
                recordFunction(null, paramsCount);
            }
        },
        FunctionExpression(path) {
            // Similar handling as arrow functions
            if (path.parent.type === 'VariableDeclarator' && path.parent.id.type === 'Identifier') {
                const name = path.parent.id.name;
                const paramsCount = path.node.params.length;
                recordFunction(name, paramsCount);
            } else if (path.parent.type === 'ClassMethod' && path.parent.key.type === 'Identifier') {
                // ClassMethod would normally be handled by ClassMethod visitor, but let's cover all bases
                const classMethodName = path.parent.key.name;
                let className = "UnknownClass";
                let current = path.parentPath;
                while (current && current.node && current.node.type !== 'ClassDeclaration') {
                    current = current.parentPath;
                }
                if (current && current.node && current.node.id && current.node.id.name) {
                    className = current.node.id.name;
                }
                const name = `${className}.${classMethodName}`;
                const paramsCount = path.node.params.length;
                recordFunction(name, paramsCount);
            } else {
                // Anonymous function expression
                const paramsCount = path.node.params.length;
                recordFunction(null, paramsCount);
            }
        },
        ClassMethod(path) {
            if (path.node.key.type === 'Identifier') {
                const methodName = path.node.key.name;
                const paramsCount = path.node.params.length;
                let className = "UnknownClass";
                let current = path.parentPath;
                while (current && current.node && current.node.type !== 'ClassDeclaration') {
                    current = current.parentPath;
                }
                if (current && current.node && current.node.id && current.node.id.name) {
                    className = current.node.id.name;
                }
                const name = `${className}.${methodName}`;
                recordFunction(name, paramsCount);
            } else {
                // Class method without a simple identifier name
                const paramsCount = path.node.params.length;
                recordFunction(null, paramsCount);
            }
        }
    });

    return functions;
}

(async function () {
    let input = "";
    process.stdin.on('data', chunk => input += chunk);
    process.stdin.on('end', () => {
        try {
            const { old, new: newCode } = JSON.parse(input);
            const oldAst = parse(old, { sourceType: "module", plugins: ["jsx", "typescript", "classProperties"] });
            const newAst = parse(newCode, { sourceType: "module", plugins: ["jsx", "typescript", "classProperties"] });

            const oldFunctions = extractFunctions(oldAst); // Map<name, {paramsCount}>
            const newFunctions = extractFunctions(newAst); // Map<name, {paramsCount}>

            let functionAdditions = 0;
            let functionRemovals = 0;
            let functionSignatureChanges = 0;

            // Check additions
            for (const [fname, fmeta] of newFunctions) {
                if (!oldFunctions.has(fname)) {
                    functionAdditions++;
                } else {
                    // Check if param count changed
                    const oldParams = oldFunctions.get(fname).paramsCount;
                    if (oldParams !== fmeta.paramsCount) {
                        functionSignatureChanges++;
                    }
                }
            }

            // Check removals
            for (const fname of oldFunctions.keys()) {
                if (!newFunctions.has(fname)) {
                    functionRemovals++;
                }
            }

            const total_changes = functionAdditions + functionRemovals + functionSignatureChanges;
            console.log(JSON.stringify({
                function_changes: total_changes,
                additions: functionAdditions,
                removals: functionRemovals,
                signature_changes: functionSignatureChanges
            }));
        } catch (err) {
            console.error(err);
            process.exit(1);
        }
    });
})();
