from metaflow import FlowSpec, step, card


class HelloFlow(FlowSpec):
    @card
    @step
    def start(self):
        """
        This is the 'start' step. All flows must have a step named 'start' that
        is the first step in the flow.
        """
        self.my_var = "hello world"
        self.next(self.a)

    @card
    @step
    def a(self):
        """
        This is a sample step that prints the var
        """
        print("the data artifact is: %s" % self.my_var)
        self.next(self.end)

    @card
    @step
    def end(self):
        """
        This is the 'end' step. All flows must have an 'end' step, which is the
        last step in the flow.
        """
        print("the data artifact is still: %s" % self.my_var)


if __name__ == "__main__":
    HelloFlow()
