##
using Plots
using Printf
##

##
# function predicting_plot(profile::String, file_name::String, model_loc::String,
#     model_type::String,  iEpoch::String,
#     Y::Array{Float64,1}, PRED::Array{Float64,1},
#     RMS::Array{Float64,1},
#     val_perf::Array{Float64,1}, TAIL::Int64,
#     save_plot::Bool=false, RMS_plot::Bool=true
#    )
function (input::String)
    println(input)    
end
function myArrayFn{T<:Number}(x::Array{T})
    println("array size: $(size(x))");
    println("max element: $(maximum(x))")
    println("min element: $(minimum(x))")
    return 2x
end
function predicting_plot(range::UnitRange{Int64}, y_pred::Array{Float64,1},
                         title_str::String)
    
    tl = @sprintf("Model %s", title_str)
    plot(range, y_pred*100,
         title = tl,
         label = ["True" "Line 2"], lw = 3)
    plot!(range, y_pred*50,
         title = tl,
         label = ["True2" "Line 2"], lw = 3)
    plot(rand(10),label="left",legend=:topleft)
    plot!(twinx(),100rand(10),color=:red,xticks=:none,label="right")
    xlabel!("Time Slice (s)")
    ylabel!("SoC (%)")
    return nothing
end

title_str = "№1"
x = 1:10; y = rand(10); # These are the plotting data 
predicting_plot(x,y,title_str)
##
plot(rand(10),label="left",legend=:topleft,fill = (0, 0.2, :red))
annotate!(9,0.75, text("mytext", :red, :right, 10))
plot!(twinx(),100rand(10),color=:red,xticks=:none,label="right")

