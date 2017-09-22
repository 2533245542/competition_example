library(shiny)

ui <- flowLayout(
  titlePanel("Competition Day"),
  sidebarLayout(
    sidebarPanel(
      selectInput(inputId = "setofdata",
                  label = "Pick a set of data:",
                  choices = c("Question1", "Question2", "Question3" )),
      numericInput(inputId = "observation",
                   label = "What observation you want to view?",
                   value = 15)
    ),
    
    mainPanel(
      verbatimTextOutput("conclusion"),
      tableOutput("look")
    )
  )
)

server <- function(input, output){
  inputDataSet <- reactive({
    switch(input$setofdata,
           "Question1" = Question1,
           "Question2" = Question2,
           "Question3" = Question3)
  })
  output$conclusion <- renderPrint({
    setofdata <- inputDataSet()
    summary(setofdata)
  })
  output$look <- renderTable({
    head(inputDataSet(), n = input$observation)
  })
}

shinyApp(ui = ui, server = server)
