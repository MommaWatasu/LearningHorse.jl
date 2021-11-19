function MV(path, forest; rounded = false, bg = "#ffffff", fc = "#000000", label = "Tree", fs = "18", nc="#ffffff") # Model Visualization
    if path[length(path) - 3 : length(path)] != ".dot"
        throw("The file path you passed is not prefixed with'dot'.")
    end
    open(path, "w") do file
        println(file, "digraph Tree_model{")
        println(file, "graph[")
        println(file, "label = \""*label*"\",")
        println(file, "bgcolor = \""*bg*"\",")
        println(file, "fontcolor = \""*fc*"\",")
        println(file, "fontsize = \""*fs*"\",")
        println(file, "style = \"filled\",")
        println(file, "margin = 0.2\n];")
        println(file, "node[")
        println(file, "shape = box,")
        if rounded
            println(file, "style = \"rounded, filled\",")
        end
        println(file, "fillcolor = \""*nc*"\"\n];")
        i = 1
        edges = []
        function tree_write(DT::DecisionTree, i; return_i = false)
            tree = DT.tree
            s = []
            push!(s, (nothing, tree))
            while length(s) != 0
                old_name, v = pop!(s)
                name = "node"*string(i)
                println(file, name*" [")
                st = ""
                if v["left"] != nothing
                    st = "X["*string(v["feature_id"])*"]<"*string(v["threshold"])*"\n"
                end
                samples = sum(v["class_count"])
                st *= "samples="*string(samples)*"\n"
                st *= "value="*string(v["class_count"])
                println(file, "label = \""*st*"\"")
                println(file, "];")
                if old_name == nothing
                elseif old_name != "node1"
                    push!(edges, old_name*"->"*name*";")
                elseif name == "node2"
                    push!(edges, "node1->node2[label = \"True\"];")
                else
                    push!(edges, "node1->"*name*"[label = \"False\"];")
                end
                i += 1
                if v["left"] != nothing
                    push!(s, (name, v["right"]))
                    push!(s, (name, v["left"]))
                end
                if return_i
                    return i
                end
            end
        end
        if typeof(forest) == DecisionTree
           tree_write(forest, i)
        else
            for tree in forest.forest
                i = tree_write(tree, i, return_i = true)
            end
        end
        for edge in edges
            println(file, edge)
        end
        print(file, "}")
    end
end