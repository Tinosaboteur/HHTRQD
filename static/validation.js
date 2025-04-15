// static/validation.js
document.addEventListener('DOMContentLoaded', () => {
    const comparisonInputs = document.querySelectorAll('input[type="number"].comparison-input');

    // Adjusted Saaty scale including exact reciprocals for comparison
    const saatyScale = [1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    const allowedTolerance = 0.01; // Allow slight deviation for reciprocals like 1/3

    comparisonInputs.forEach(input => {
        const messageSpan = input.nextElementSibling; // Assuming span is right after input

        // Function to find the closest Saaty value (optional, for better suggestions)
        function findClosestSaaty(value) {
            return saatyScale.reduce((prev, curr) => {
                return (Math.abs(curr - value) < Math.abs(prev - value) ? curr : prev);
            });
        }

        input.addEventListener('input', (event) => {
            validateInput(event.target, messageSpan);
        });
        // Also validate on change (e.g., when user clicks away)
        input.addEventListener('change', (event) => {
            validateInput(event.target, messageSpan);
        });

        // Initial validation check on load (e.g., if form was repopulated after error)
        if (input.value) {
             validateInput(input, messageSpan);
        }
    });

    function validateInput(inputElement, messageSpan) {
        messageSpan.textContent = ''; // Clear previous message
        inputElement.setCustomValidity(''); // Clear previous custom validity
        inputElement.style.borderColor = ''; // Reset border color
        inputElement.style.backgroundColor = ''; // Reset background color


        if (!inputElement.value) {
            // 'required' attribute will handle empty on submit, but we might want immediate feedback
            // messageSpan.textContent = 'Giá trị là bắt buộc.';
            // inputElement.setCustomValidity('Giá trị là bắt buộc.');
            return; // Don't validate further if empty
        }

        try {
            const value = parseFloat(inputElement.value);

            if (isNaN(value)) {
                messageSpan.textContent = 'Vui lòng nhập một số.';
                inputElement.setCustomValidity('Định dạng số không hợp lệ.');
                 inputElement.style.borderColor = 'var(--error-color)';
                return;
            }

            // Check basic bounds (must be positive)
            // AHP requires values > 0. Saaty scale is 1/9 to 9.
            // Let's enforce slightly wider bounds to allow inputs like 0.1 or 10,
            // but warn about Saaty scale.
             if (value <= 0) {
                messageSpan.textContent = 'Giá trị phải lớn hơn 0.';
                inputElement.setCustomValidity('Giá trị phải là số dương.');
                 inputElement.style.borderColor = 'var(--error-color)';
                return;
            }

            // Check if the value is "close enough" to a Saaty scale value or its reciprocal
            let isNearSaaty = saatyScale.some(scaleValue => Math.abs(value - scaleValue) < allowedTolerance);
            let isWithinBounds = (value >= (1/9 - allowedTolerance) && value <= (9 + allowedTolerance));

            if (!isWithinBounds) {
                 messageSpan.textContent = `Cảnh báo: Giá trị (${value.toFixed(2)}) nằm ngoài thang đo Saaty (1/9 ≈ 0.11 đến 9).`;
                 // Don't make it invalid, just warn
                 inputElement.style.borderColor = 'var(--warning-color)';

            } else if (!isNearSaaty) {
                 // It's not strictly invalid, but provide a warning/suggestion
                 const closest = findClosestSaaty(value);
                 let suggestion = `Gợi ý: Nên dùng giá trị thang Saaty (1-9 hoặc nghịch đảo 1/2..1/9). Giá trị gần nhất là ${closest.toFixed(2)}.`;
                 if (closest === 1/2 || closest === 1/3 || closest === 1/4 || closest === 1/5 || closest === 1/6 || closest === 1/7 || closest === 1/8 || closest === 1/9) {
                     suggestion = `Gợi ý: Nên dùng giá trị thang Saaty (1-9 hoặc nghịch đảo 1/2..1/9). Giá trị gần nhất là 1/${Math.round(1/closest)}.`;
                 }
                 messageSpan.textContent = suggestion;
                 messageSpan.style.color = '#856404'; // Warning color text
                 inputElement.style.borderColor = ''; // Clear specific warning border if message is enough
            } else {
                 messageSpan.textContent = ''; // Clear suggestion if value is near Saaty
                 messageSpan.style.color = ''; // Reset color
            }


        } catch (e) {
            // This catch might be redundant due to isNaN check, but safe to keep
            messageSpan.textContent = 'Vui lòng nhập một số hợp lệ.';
            inputElement.setCustomValidity('Định dạng số không hợp lệ.');
             inputElement.style.borderColor = 'var(--error-color)';
        }
    }

    // Add form submission validation to ensure all fields passed custom checks
    const forms = document.querySelectorAll('form'); // Select all forms on the page
    forms.forEach(form => {
        // Check if the form contains comparison inputs before adding listener
         if (form.querySelector('input[type="number"].comparison-input')) {
            form.addEventListener('submit', (event) => {
                let formIsValid = true;
                let firstInvalidInput = null;

                form.querySelectorAll('input[type="number"].comparison-input').forEach(input => {
                    // Re-run our custom validation logic on each input before submit
                    validateInput(input, input.nextElementSibling);

                    // Check the browser's validity state AND our custom validity message
                    if (!input.validity.valid || input.validationMessage !== '') {
                        formIsValid = false;
                        if (!firstInvalidInput) {
                            firstInvalidInput = input;
                        }
                        // Ensure visual indication persists
                        input.style.borderColor = 'var(--error-color)';
                        input.style.backgroundColor = '#fffafa';
                    }
                });

                if (!formIsValid) {
                    console.log("Form submission prevented due to validation errors.");
                    event.preventDefault(); // Stop form submission
                    if (firstInvalidInput) {
                        firstInvalidInput.focus(); // Focus the first invalid field
                        // Scroll to the element smoothly if possible
                        firstInvalidInput.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                    // Use flash message area or a dedicated area for a general error message
                    // Find or create an error message container
                    let errorContainer = form.querySelector('.form-submit-error');
                    if (!errorContainer) {
                        errorContainer = document.createElement('div');
                        errorContainer.className = 'alert alert-error form-submit-error'; // Use alert styles
                        // Prepend error message within the form, before the button container perhaps
                        const buttonContainer = form.querySelector('.button-container');
                         if (buttonContainer) {
                            form.insertBefore(errorContainer, buttonContainer);
                         } else { // Fallback append if no button container found
                              form.appendChild(errorContainer);
                         }
                    }
                    errorContainer.textContent = 'Vui lòng sửa các lỗi được đánh dấu trong biểu mẫu trước khi tiếp tục.';
                    errorContainer.style.display = 'block'; // Make sure it's visible

                } else {
                     // Clear any previous general form error message if the form is now valid
                     let errorContainer = form.querySelector('.form-submit-error');
                      if (errorContainer) {
                          errorContainer.style.display = 'none';
                      }
                }
            });
         } // End if form contains comparison inputs
    }); // End forms.forEach

}); // End DOMContentLoaded